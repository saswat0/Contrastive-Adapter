"""
Functions for working with a CLIP model
"""  
import os
import numpy as np
import torch

import umap

from tqdm import tqdm
from sklearn.cluster import KMeans

from clip import clip

from utils.logging import summarize_acc



def get_embeddings(text, clip_model, args, normalize=True, verbose=True):
    if verbose:
        desc = '-> Text descriptions for zero-shot classification:'
        print('-' * len(desc))
        print(desc)
        num_display = 5
        for d in text[:num_display]:
            print(f'   - {d}')
        if len(text) > num_display:
            print(f'     ...')
            for d in text[-num_display:]:
                print(f'   - {d}')
        print('-' * len(desc))
    if 'clip' in args.load_base_model or 'cloob' in args.load_base_model:
        text_tokens = clip.tokenize(text)
    elif 'slip' in args.load_base_model:
        slip_tokenizer = SLIPSimpleTokenizer()
        text_tokens = slip_tokenizer(text)
        text_tokens = text_tokens.view(-1, 77).contiguous()
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        if normalize:
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    clip_model.cpu()
    return text_embeddings


def get_dataset_embeddings(model, dataloader, args, split='train'):
    return get_clip_embeddings(model, dataloader, args, split)


def get_clip_embeddings(model, dataloader, args, 
                        split='train', verbose=False):
    verbose = True if args.verbose else False

    dataset = args.dataset.replace('_iid', '').split('_min')[0]
    args.embeddings_dir = '../embeddings/'
    args.embeddings_dir = os.path.join(args.embeddings_dir, args.dataset, args.config)
    embedding_fname = f'd={dataset}-s={split}-c={args.config}-m={args.load_base_model}.pt'
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
    try:
        if os.path.exists(embedding_path):
            if verbose:
                print(f'-> Retrieving image embeddings from {embedding_path}!')
            embeddings = torch.load(embedding_path)

            return embeddings
        else:
            if verbose:
                print(f'-> Image embeddings from {embedding_path} not found.')
                
    except:
        pass
    
    model.to(args.device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for ix, data in enumerate(tqdm(dataloader, 
                                       desc=f'Computing {args.load_base_model} image embeddings for {split} split')):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            try:
                embeddings = model.encode_image(inputs).float().cpu()
                all_embeddings.append(embeddings)
                inputs = inputs.cpu()
                labels = labels.cpu()
            except Exception as e:
                import pdb; pdb.set_trace()
    model.cpu()
    
    # Save to disk
    torch.save(torch.cat(all_embeddings), embedding_path)
    if verbose:
        print(f'-> Saved image embeddings to {embedding_path}!')
    
    return torch.cat(all_embeddings)


def get_clip_text_embeddings(model, dataloader, args, 
                             split='train'):
    # If already computed, retrieve from disk
    if 'breeds' in args.dataset:
        dataset = 'imagenet'
        embeddings_dir = '../embeddings/imagenet'
        if not os.path.exists(embeddings_dir):
            os.mkdir(embeddings_dir)
    else:
        dataset = args.dataset
        embeddings_dir = args.embeddings_dir
    
    embedding_fname = f'd={dataset}-s={split}-c={args.config}-m={args.load_base_model}.pt'
    embedding_path = os.path.join(embeddings_dir, embedding_fname)
    if os.path.exists(embedding_path):
        print(f'-> Retrieving text embeddings from {embedding_path}!')
        return torch.load(embedding_path)
    else:
        print(f'-> Text embeddings from {embedding_path} not found.')
    
    model.to(args.device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for ix, data in enumerate(tqdm(dataloader, 
                                       desc=f'Computing {args.load_base_model} text embeddings')):
            inputs, labels, data_ix = data
            # Inputs should be tokenized by CLIP first.
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            embeddings = model.encode_text(inputs).float().cpu()
            all_embeddings.append(embeddings)
            inputs = inputs.cpu()
            labels = labels.cpu()
    
    # Save to disk
    torch.save(torch.cat(all_embeddings), embedding_path)
    print(f'-> Saved text embeddings to {embedding_path}!')
    
    return torch.cat(all_embeddings)
        
    
def get_umap_embeddings(embeddings, seed, split='train', args=None):
    # If already computed, retrieve from disk
    embedding_fname = f'umap-d={args.dataset}-split={split}-c={args.config}-m={args.load_base_model}-s={seed}.npy'
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
    if os.path.exists(embedding_path):
        print(f'-> Retrieving UMAP embeddings from {embedding_path}!')
        return np.load(embedding_path)
    else:
        print(f'-> UMAP embeddings from {embedding_path} not found.')
        print(f'-> Computing UMAP of embedding size {[s for s in embeddings.shape]}...')
        
        embeddings = umap.UMAP(random_state=seed).fit_transform(embeddings)
        np.save(embedding_path, embeddings)
        return embeddings
        
        
    
def evaluate_clip(clip_predictions, dataloader, verbose=False):
    """
    General method for classification validation
    Args:
    - clip_predictions (np.array): predictions
    - dataloader (torch.utils.data.DataLoader): (unshuffled) dataloader
    """
    targets_t = dataloader.dataset.targets_all['target'].astype(int)
    targets_s = dataloader.dataset.targets_all['spurious'].astype(int)

    correct_by_groups = np.zeros([len(np.unique(targets_t)),
                                  len(np.unique(targets_s))])
    auroc_by_groups = np.zeros([len(np.unique(targets_t)),
                                len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)

    correct = (clip_predictions == targets_t)
    
    for ix, y in enumerate(targets_t):
        s = targets_s[ix]
        correct_by_groups[int(y)][int(s)] += correct[ix].item()
        total_by_groups[int(y)][int(s)] += 1
        
    avg_acc, robust_acc = summarize_acc(correct_by_groups,
                                        total_by_groups,
                                        stdout=verbose)
    return avg_acc, robust_acc


def classify_embeddings_with_text(clip_model,
                                  image_embeddings, 
                                  text_descriptions,
                                  args,
                                  temperature=100.):
    """
    Example: 
    text_descriptions = [f"This is a photo of a {label}." 
                           for label in ['land bird', 'water bird']]
    """
    text_tokens = clip.tokenize(text_descriptions)
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        _image_embeddings = (image_embeddings / 
                             image_embeddings.norm(dim=-1, keepdim=True))
        
        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        _text_embeddings = (text_embeddings / 
                            text_embeddings.norm(dim=-1, keepdim=True))
        
        cross = _image_embeddings @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)
        
        text_tokens = text_tokens.cpu()
        
    clip_model.cpu()
    return predicted.cpu().numpy()



def classify_embeddings_by_prototype(image_embeddings, 
                                     dataloader,
                                     args,
                                     temperature=100.):
    targets = dataloader.dataset.targets
    class_indices = [np.where(targets == t)[0] 
                     for t in np.unique(targets)]
    class_embeddings = torch.stack([image_embeddings[i].mean(dim=0)
                                    for i in class_indices])
    
    with torch.no_grad():
        _image_embeddings = (image_embeddings / 
                             image_embeddings.norm(dim=-1, keepdim=True))
        _class_embeddings = (class_embeddings /
                             class_embeddings.norm(dim=-1, keepdim=True))
        
        cross = _image_embeddings @ _class_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)
        
    return predicted.cpu().numpy()


def classify_embeddings_with_clustering(embeddings, 
                                        labels,
                                        num_classes, 
                                        seed):
    """
    image_embeddings can be dimensionality-reduced via UMAP or direct CLIP embeddings
    """
    kmeans = KMeans(n_clusters=num_classes,
                    random_state=seed)
    kmeans.fit(embeddings)
    predicted = kmeans.labels_
    # 2-class heurstic for now
    correct_0 = (predicted == labels).sum()
    correct_1 = (1 - predicted == labels).sum()
    if correct_1 > correct_0:
        predicted = 1 - predicted
    return predicted


def get_zeroshot_predictions(key_embeddings,
                             text_descriptions,
                             predict_by,
                             args,
                             dataloader,
                             temperature=100.,
                             split='train',
                             numpy=True,
                             base_model=None):
    if predict_by == 'text':
        predictions = classify_embeddings_with_text(
            base_model, key_embeddings, text_descriptions, args,
            temperature=100.
        )
        
    elif predict_by == 'ground_truth':
        predictions = dataloader.dataset.targets_all['spurious']
    elif predict_by == 'proto':
        predictions = classify_embeddings_by_prototype(key_embeddings, 
                                                       dataloader,
                                                       args,
                                                       temperature)
    else:
        if 'umap' in predict_by:
            print(f'Computing UMAP...')
            key_embeddings = get_umap_embeddings(key_embeddings,
                                                 seed=args.seed,
                                                 split=split,
                                                 args=args)
        predictions = classify_embeddings_with_clustering(
            key_embeddings, dataloader.dataset.targets_all['target'],
            args.num_classes, args.seed)
        
    return predictions
        


def get_clip_predictions(image_embeddings, 
                         dataloader, 
                         clip_model, 
                         args, 
                         clip_predict_by=None,
                         split='train',
                         numpy=True):
    # clip_predict_by should override args.clip_predict_by
    if args.clip_predict_by == 'text' or clip_predict_by == 'text':
        # Hacks
        if args.dataset in ['waterbirds', 'celebA']:
            text_descriptions = args.text_descriptions
            clip_predictions = classify_embeddings_with_text(
                clip_model, image_embeddings, args.text_descriptions, args,
                temperature=100.)
        elif args.dataset == 'fmow':
            try:
                zeroshot_weights = args.zeroshot_weights
            except:
                zeroshot_weights = clip_zeroshot_classifier(clip_model,
                                                            classnames=args.train_classes, 
                                                            templates=args.prompt_templates,
                                                            args=args)
                args.zeroshot_weights = zeroshot_weights
            clip_predictions = classify_embeddings_with_zeroshot_weights(image_embeddings, 
                                                                         zeroshot_weights,
                                                                         args,
                                                                         temperature=100.,
                                                                         numpy=False)
            if numpy:
                clip_predictions = clip_predictions.numpy()
            
        else:
            raise NotImplementedError
        
    elif args.clip_predict_by == 'ground_truth' or clip_predict_by == 'ground_truth':
        clip_predictions = dataloader.dataset.targets_all['spurious']
    else:
        if 'umap' in args.clip_predict_by or 'umap' in clip_predict_by:
            print(f'Computing UMAP...')
            image_embeddings = get_umap_embeddings(image_embeddings,
                                                   seed=args.seed,
                                                   split=split,
                                                   args=args)
        clip_predictions = classify_embeddings_with_clustering(
            image_embeddings, dataloader.dataset.targets_all['target'],
            args.num_classes, args.seed)
        
    return clip_predictions


def clip_zeroshot_classifier(clip_model, classnames, templates, args):
    clip_model.to(args.device)
    clip_model.eval()
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc='Computing CLIP zero-shot embedding vectors'):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(args.device) # tokenize
            class_embeddings = clip_model.encode_text(texts) # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def classify_embeddings_with_zeroshot_weights(image_embeddings, 
                                                    zeroshot_weights,
                                                    args,
                                                    temperature=100.,
                                                    numpy=False):
    """
    Example: 
    text_descriptions = [f"This is a photo of a {label}." 
                           for label in ['land bird', 'water bird']]
    """
    with torch.no_grad():  
        image_embeddings = image_embeddings.to(args.device)
        cross = image_embeddings @ zeroshot_weights.float().to(args.device)
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)
        
        image_embeddings = image_embeddings.cpu()
        zeroshot_weights = zeroshot_weights.cpu()
        predicted = predicted.cpu()
        
    if numpy:
        return predicted.numpy()
    return predicted
