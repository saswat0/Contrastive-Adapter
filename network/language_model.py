"""
Functions for language foundation models
"""
import os
import numpy as np
import torch

from tqdm import tqdm

from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer
from transformers import pipeline, FeatureExtractionPipeline
from transformers.pipelines.pt_utils import KeyDataset

# Should refactor this
from network.clip import get_umap_embeddings, classify_embeddings_with_clustering


def return_supported_language_models():
    supported_models = ['facebook/bart-large-mnli',
                        'EleutherAI/gpt-neo-2.7B',
                        'EleutherAI/gpt-neo-1.3B',
                        'EleutherAI/gpt-neo-125M',
                        'roberta-large',
                        'roberta-large-mnli']
    for m in supported_models:
        print(f'- {m}')
    return supported_models


def get_embeddings(text, pipeline, args, normalize=True, verbose=True):
    try: 
        assert len(text[0]) > 1
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
            
        tokenizer_kwargs = {'padding':False, 
                            'truncation':True,
                            'max_length':args.max_token_length,
                           }
        embeddings = pipeline(list(text),
                              **tokenizer_kwargs)
        # Compute mean feature values
        embeddings = torch.vstack([torch.tensor(e[0]).float().cpu().mean(dim=0) 
                                   for e in embeddings])
        if normalize:
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    except:
        raise TypeError


def get_dataset_embeddings(pipeline, dataloader, args, split='train'):
    base_model = args.load_base_model.replace('/', '_').replace('-', '_')
    
    embedding_fname = f'd={args.dataset}-s={split}-c={args.config}-mt={args.max_token_length}-m={base_model}.pt'
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
    
    if os.path.exists(embedding_path):
        print(f'-> Retrieving {args.load_base_model} embeddings from {embedding_path}!')
        embeddings = torch.load(embedding_path)
        print(f'   - {split} embeddings shape: {embeddings.shape}')
        return embeddings  # orch.load(embedding_path)
    else:
        print(f'-> {args.load_base_model} embeddings from {embedding_path} not found. Computing.')

    tokenizer_kwargs = {'device': args.device, 
                        'truncation': True,
                        'max_length': args.max_token_length,
                        'return_tensors': 'pt'}
        
    all_embeddings = []
    with torch.no_grad():
        for data in tqdm(pipeline(KeyDataset(dataloader.dataset, 0), **tokenizer_kwargs),
                         desc=f'Computing {args.load_base_model} embeddings for {split} split',
                         total=len(dataloader.dataset)):
            embeddings = torch.tensor(data[0][0]).float().cpu()
            all_embeddings.append(embeddings)
    torch.save(torch.vstack(all_embeddings), embedding_path)
    print(f'-> Saved {args.load_base_model} embeddings to {embedding_path}!')
    return torch.cat(all_embeddings)


def classify_embeddings_with_text(key_embeddings, 
                                  query_embeddings,
                                  temperature=100.,
                                  normalize_query=False):
    """
    Example: 
    text_descriptions = [f"This is a photo of a {label}." 
                           for label in ['land bird', 'water bird']]
    """
    with torch.no_grad():
        _key_embeddings = (key_embeddings / 
                           key_embeddings.norm(dim=-1, keepdim=True))
        if normalize_query is True:
            _query_embeddings = (query_embeddings / 
                                 query_embeddings.norm(dim=-1, keepdim=True))
        else:
            _query_embeddings = query_embeddings
        
        cross = _key_embeddings @ _query_embeddings.T
        probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(probs.data, 1)

    return predicted.cpu()


def get_zeroshot_predictions(key_embeddings, 
                             query_embeddings,
                             predict_by,
                             args,
                             dataloader,
                             temperature=100.,
                             split='train',
                             numpy=True,
                             base_model=None):
    if predict_by == 'text':
        predictions = classify_embeddings_with_text(key_embeddings, 
                                                    query_embeddings,
                                                    temperature)
        print(np.unique(predictions, return_counts=True))
        return predictions.numpy() if numpy else predictions
    elif predict_by == 'ground_truth':
        return dataloader.dataset.targets_all['spurious']
    
    else:
        if 'umap' in predict_by:
            print(f'Computing UMAP...')
            embeddings = get_umap_embeddings(key_embeddings,
                                             seed=args.seed,
                                             split=split,
                                             args=args)
        return classify_embeddings_with_clustering(
            embeddings, dataloader.dataset.targets_all['target'], 
            args.num_classes, args.seed)
    
    
class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def __init__(self, 
                 model, 
                 tokenizer,
                 padding='max_length', 
                 truncation=True, 
                 max_token_length=300):  # for CivilComments
        super().__init__(model=model, tokenizer=tokenizer)
        self.padding = padding
        self.truncation = truncation
        self.max_token_length = max_token_length
        
    def preprocess(self, inputs, truncation=True):
        return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, 
                                      padding=self.padding,
                                      truncation=self.truncation,
                                      max_length=self.max_token_length,
                                      return_tensors=return_tensors)
        return model_inputs
        
        

def load_pipeline(model, tokenizer, max_token_length, device=-1, 
                  task='feature-extraction'):
    return pipeline('feature-extraction', model=model,
                    tokenizer=tokenizer, device=device)


def load_pretrained_language_model(sequence_classification, args):
    """
    Returns model and transform for a Huggingface pretrained model
    """
    if 'cls' in args.load_base_model:
        load_base_model = args.load_base_model[:-4]
    else:
        load_base_model = args.load_base_model
    print(f'-> Loading {load_base_model} from 🤗 models.')
    
    if 'gpt-neo' in args.load_base_model:
        args.bos_token = '<|endoftext|>'
        args.eos_token = '<|endoftext|>'
        args.pad_token = '<|pad|>'
        tokenizer = GPT2Tokenizer.from_pretrained(load_base_model,
                                                  cache_dir=args.cache_dir,
                                                  truncation=True,
                                                  model_max_length=args.max_token_length,
                                                  max_length=args.max_token_length,
                                                  return_tensors='pt'
                                                  )
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_base_model,
                                                  cache_dir=args.cache_dir,
                                                  padding='max_length',
                                                  truncation=True,
                                                  model_max_length=args.max_token_length,
                                                  max_length=args.max_token_length,
                                                  return_tensors='pt')
    if sequence_classification:
        model = AutoModelForSequenceClassification.from_pretrained(load_base_model,
                                                                   cache_dir=args.cache_dir,
                                                                   output_hidden_states=True)
        transform = lambda text: transform_for_binary_sequence_classification(text, 
                                                                              tokenizer,
                                                                              args)
    else:
        model = AutoModel.from_pretrained(load_base_model,
                                          cache_dir=args.cache_dir,
                                          output_hidden_states=True)
        if 'gpt-neo' in args.load_base_model:
            transform = lambda text: transform_for_gpt_neo(text,
                                                           tokenizer,
                                                           args)
        else:
            transform = lambda text: transform_for_base_model(text, 
                                                              tokenizer, 
                                                              args)
    return model, transform, tokenizer


def transform_for_binary_sequence_classification(text, tokenizer, args):
    tokens = tokenizer.encode(text,
                              args.text_descriptions[1],
                              padding='max_length',
                              truncation=True,
                              max_length=args.max_token_length,
                              return_tensors='pt')
    return tokens


def transform_for_base_model(text, tokenizer, args):
    tokens = tokenizer([text] + args.text_descriptions, 
                       padding='max_length',
                       truncation=True,
                       max_length=args.max_token_length,
                       return_tensors='pt')
    return tokens


def transform_for_gpt_neo(text, tokenizer, args):
    tokens = tokenizer([f'{args.bos_token}text{args.eos_token}'] + 
                       [f'{args.bos_token}{text}{args.eos_token}'
                        for text in args.text_descriptions],
                       padding=False,
                       truncation=True,
                       max_length=args.max_token_length,
                       return_tensors='pt')
    return tokens
    
