"""
Datasets and data utils

Functions:
- initialize_data()
- train_val_split()
- get_resampled_set()
- imshow()
- plot_data_batch()
"""
import copy
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image


def initialize_data(args):
    """
    Set dataset-specific default arguments
    """
    args.resample_iid = False
    if 'split_cifar100' in args.dataset:
        dataset = '_'.join(args.dataset.split('_')[:-1])
    elif 'cifar10e' in args.dataset:
        dataset = 'cifar10e'
    else:
        dataset = args.dataset.split('_')[0]
    dataset_module = importlib.import_module(f'datasets.{dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')
    visualize_dataset = getattr(dataset_module, 'visualize_dataset')
    
    args.results_dict = {'epoch': [],
                         'dataset_ix': [],  # legacy
                         'train_loss': [],
                         'train_avg_acc': [],
                         'train_robust_acc': [],
                         'val_loss': [],
                         'val_avg_acc': [],
                         'val_robust_acc': [],
                         'test_loss': [],
                         'test_avg_acc': [],
                         'test_robust_acc': [],
                         'best_loss_epoch': [],
                         'best_acc_epoch': [],
                         'best_robust_acc_epoch': []}
        
    if args.dataset == 'waterbirds':
        # Update this to right path
        args.root_dir = './datasets/data/Waterbirds/' 
        args.val_split = 0.2
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.augment_data = False
        args.train_classes = ['landbird', 'waterbird']
        ## Image
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.text_descriptions = ['a landbird', 'a waterbird']
        args.wilds_dataset = False
        
    elif 'celebA' in args.dataset:
        args.root_dir = './datasets/data/CelebA/'
        # IMPORTANT - dataloader assumes that we have directory structure
        # in ./datasets/data/CelebA/ :
        # |-- list_attr_celeba.csv
        # |-- list_eval_partition.csv
        # |-- img_align_celeba/
        #     |-- image1.png
        #     |-- ...
        #     |-- imageN.png
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = './images/celebA/'
        args.train_classes = ['nonblond', 'blond']
        args.val_split = 0.2
        args.text_descriptions = ['a celebrity with dark hair',
                                  'a celebrity with blond hair']
        args.wilds_dataset = False
        
        
    elif args.dataset == 'cifar10e1' or args.dataset == 'cifar10e2':
        args.root_dir = './datasets/data/cifar10'
        args.target_name = 'class'
        args.confounder_names = ['dataset']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = f'./images/{args.dataset}/'
        args.train_classes = ['airplane',
                              'automobile',
                              'bird',
                              'cat',
                              'deer',
                              'dog',
                              'frog',
                              'horse',
                              'ship',
                              'truck']
        args.prompt_templates = [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a black and white photo of a {}.',
            'a low contrast photo of a {}.',
            'a high contrast photo of a {}.',
            'a bad photo of a {}.',
            'a good photo of a {}.',
            'a photo of a small {}.',
            'a photo of a big {}.',
            'a photo of the {}.',
            'a blurry photo of the {}.',
            'a black and white photo of the {}.',
            'a low contrast photo of the {}.',
            'a high contrast photo of the {}.',
            'a bad photo of the {}.',
            'a good photo of the {}.',
            'a photo of the small {}.',
            'a photo of the big {}.',
        ]
        args.val_split = 0.2
        args.text_descriptions = [f'an {c}' if c[0] in ['a', 'e', 'i', 'o', 'u']
                                  else f'a {c}' for c in args.train_classes]
        args.wilds_dataset = True 
        args.worst_group_key = 'acc_wg'
        args.average_group_key = 'acc_avg'
        
        
    elif 'civilcomments' in args.dataset:
        args.root_dir = './datasets/data/WILDS_datasets/'
        args.target_name = 'toxic'
        args.confounder_names = ['identities']
        args.image_mean = 0
        args.image_std = 0
        args.augment_data = False
        args.image_path = './images/civilcomments/'
        args.train_classes = ['not toxic', 'toxic']
        args.max_token_length = 300
        args.wilds_dataset = True
        args.worst_group_key = 'acc_wg'
        args.average_group_key = 'acc_avg'
        # Prompt-tuned
        args.text_descriptions = ['Not toxic.', 'Toxic.']

    elif 'amazon' in args.dataset:
        args.root_dir = './datasets/data/WILDS_datasets/'
        args.target_name = 'positive'
        args.confounder_names = [args.dataset.split('_')[-1]]
        args.image_mean = 0
        args.image_std = 0
        args.augment_data = False
        args.image_path = './images/amazon/'
        args.train_classes = ['negative', 'positive']
        args.max_token_length = 512
        args.wilds_dataset = True
        args.worst_group_key = 'acc_wg'
        args.average_group_key = 'acc_avg'
        if 'user' in args.dataset:
            args.worst_group_key = '10th_percentile_acc'

        # Prompt-tuned
        args.text_descriptions = ['negative', 'positive']
        args.prompt_templates = ['{}'] 
        args.prompts = ['negative review', 
                        'positive review']

    elif 'breeds' in args.dataset:
        args.root_dir = './datasets/data/BREEDS'
        args.target_name = 'super'
        args.confounder_names = ['sub']
        args.augment_data = False
        args.image_path = f'./images/{args.dataset}/'
        args.wilds_dataset = True
        args.worst_group_key = 'acc_wg'
        args.average_group_key = 'acc_avg'
        data_ids = args.dataset.split('_i')[-1].split('_')[0]
        args.resample_iid = (data_ids == 'id')
        data_ids = args.dataset.split('_')
        data_ids = args.dataset.split('_i')
        data_ids = data_ids[0].split('_')
        
        args.breeds_dataset_name = data_ids[1]
        args.breeds_subclasses = data_ids[2:]
        
        print('args.breeds_dataset_name', args.breeds_dataset_name)
        print('args.breeds_subclasses', args.breeds_subclasses)
        
    elif 'fmow' in args.dataset:
        args.root_dir = './datasets/data/WILDS_datasets/'
        args.target_name = 'building_land'
        args.confounder_names=['year', 'region']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.wilds_dataset = True
        args.worst_group_key = 'acc_worst_region'
        args.average_group_key = 'acc_avg'
        args.train_classes = ["airport", "airport hangar", "airport terminal", "amusement park", "aquaculture", "archaeological site", "barn", "border checkpoint", "burial site", "car dealership", "construction site", "crop field", "dam", "debris or rubble", "educational institution", "electric substation", "factory or powerplant", "fire station", "flooded road", "fountain", "gas station", "golf course", "ground transportation station", "helipad", "hospital", "impoverished settlement", "interchange", "lake or pond", "lighthouse", "military facility", "multi-unit residential", "nuclear powerplant", "office building", "oil or gas facility", "park", "parking lot or garage", "place of worship", "police station", "port", "prison", "race track", "railway bridge", "recreational facility", "road bridge", "runway", "shipyard", "shopping mall", "single-unit residential", "smokestack", "solar farm", "space facility", "stadium", "storage tank", "surface mine", "swimming pool", "toll booth", "tower", "tunnel opening", "waste disposal", "water treatment facility", "wind farm", "zoo"]
        args.train_classes = [c.replace('_', ' ') for c in 
                              args.train_classes]
        args.text_descriptions = [f'an aerial view of the {c}' 
                                  for c in args.train_classes] 
 
        args.prompt_templates = ['satellite imagery of {}.', 
                                 'aerial imagery of {}.', 
                                 'satellite photo of {}.',
                                 'an aerial photo of {}.',
                                 'satellite view of {}.',
                                 'aerial view of {}.',
                                 'satellite imagery of a {}.',
                                 'aerial imagery of a {}.',
                                 'a satellite photo of a {}.',
                                 'an aerial photo of a {}.',
                                 'a satellite view of a {}.',
                                 'an aerial view of a {}.',
                                 'satellite imagery of the {}.',
                                 'aerial imagery of the {}.',
                                 'a satellite photo of the {}.',
                                 'an aerial photo of the {}.',
                                 'a satellite view of the {}.',
                                 'an aerial view of the {}.']
        args.save_id = 'iid'
        args.results_dict[f'val_{args.save_id}_loss'] = []
        args.results_dict[f'val_{args.save_id}_avg_acc'] = []
        args.results_dict[f'val_{args.save_id}_robust_acc'] = []
        args.results_dict[f'test_{args.save_id}_loss'] = []
        args.results_dict[f'test_{args.save_id}_avg_acc'] = []
        args.results_dict[f'test_{args.save_id}_robust_acc'] = []

        args.results_dict[f'best_loss_{args.save_id}_epoch'] = []
        args.results_dict[f'best_acc_{args.save_id}_epoch'] = []
        args.results_dict[f'best_robust_acc_{args.save_id}_epoch'] = []
        
    else:
        raise NotImplementedError
    
    args.task = args.dataset 
    try:
        args.num_classes = len(args.train_classes)
    except:
        pass
    return load_dataloaders, visualize_dataset


def update_classification_prompts(args):        
    if args.dataset == 'civilcomments':          
        if args.load_base_model == 'EleutherAI/gpt-neo-1.3B':
            args.text_descriptions = ['Not toxic', 'Toxic']

        elif args.load_base_model == 'EleutherAI/gpt-neo-125M':
            args.text_descriptions = ['Not toxic', 'Toxic']
        elif 'EleutherAI' in args.load_base_model:
            args.text_descriptions = ['Not toxic', 'Toxic']


    elif 'amazon' in args.dataset:
        if args.load_base_model == 'EleutherAI/gpt-neo-125M':
            args.text_descriptions = ['Negative review', 'Positive review']
        elif args.load_base_model == 'EleutherAI/gpt-neo-1.3B':
            args.text_descriptions = ['Negative', 'Positive']
        elif args.load_base_model == 'EleutherAI/gpt-neo-2.7B':
            args.text_descriptions = ['Negative', 'Positive']

    elif 'breeds_nonliving26' in args.dataset and 'cloob' in args.load_base_model:
        args.text_descriptions = [f'an {c}' 
                                  if c[0] in 'aeiou' 
                                  else f'a {c}'
                                  for c in args.train_classes]
            
    elif 'breeds_nonliving26' in args.dataset and 'clip' in args.load_base_model:
        args.text_descriptions = [f'A photo of an {c}.' 
                                  if c[0] in 'aeiou' 
                                  else f'A photo of a {c}.'
                                  for c in args.train_classes]
        
    elif 'breeds_living17' in args.dataset and 'cloob' in args.load_base_model:
        args.text_descriptions = [f'This is a picture of an {c}.' 
                                  if c[0] in 'aeiou' 
                                  else f'This is a picture of a {c}.'
                                  for c in args.train_classes]
        
    elif 'breeds_living17' in args.dataset and 'clip' in args.load_base_model:
        args.text_descriptions = [f'This is a picture of an {c}.' 
                                  if c[0] in 'aeiou' 
                                  else f'This is a picture of a {c}.'
                                  for c in args.train_classes]
    
    elif 'fmow' in args.dataset and 'clip' in args.load_base_model:
        args.text_descriptions = [f'satellite view of the {c}.'
                                  for c in args.train_classes]
        
    elif 'fmow' in args.dataset and 'cloob' in args.load_base_model:
        args.text_descriptions = [f'aerial view of an {c}.' 
                                  if c[0] in 'aeiou' 
                                  else f'aerial view of a {c}.'
                                  for c in args.train_classes]
        
    elif 'waterbirds' in args.dataset and 'cloob' in args.load_base_model:
        args.text_descriptions = [f'a {c}' 
                                  if c[0] in 'aeiou' 
                                  else f'a {c}'
                                  for c in args.train_classes]
        
    elif 'waterbirds' in args.dataset and 'clip' in args.load_base_model:
        args.text_descriptions = [f'This is a picture of a {c}.' 
                                  if c[0] in 'aeiou' 
                                  else f'This is a picture of a {c}.'
                                  for c in args.train_classes]
    elif 'cifar10e' in args.dataset and 'cloob' in args.load_base_model:
        args.text_descriptions = [f'an {c}' 
                                  if c[0] in 'aeiou' 
                                  else f'a {c}'
                                  for c in args.train_classes]
        
    elif 'celebA' in args.dataset: 
        args.text_descriptions = [
            'A photo of a celebrity with dark hair.', 
            'A photo of a celebrity with blond hair.'
        ]
        
    else:
        args.text_descriptions = args.text_descriptions
    return args.text_descriptions


# ---------------------------------
# Helper functions to organize data
# ---------------------------------
def get_indices_by_label(dataloader, label_type='target'):
    indices_by_label = []
    targets = dataloader.dataset.targets_all[label_type]
    for t in np.unique(targets):
        indices_by_label.append(np.where(targets == t)[0])
    return indices_by_label

# More general than the above
def get_indices_by_value(values):
    indices = []
    for v in np.unique(values):
        indices.append(np.where(values == v)[0])
    return indices
    
def get_correct_indices(predictions, targets):
    correct = predictions == targets
    correct_indices = np.where(correct == 1)[0]
    return correct_indices, correct


def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    """
    Obtain a resampled dataset given sampling indices
    Args:
    - dataset (torch.utils.data.Dataset): Dataset
    - resampled_set_indices (int[]): Sampling indices 
    - deepcopy (bool): If true, copy the dataset
    """
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:  # Some dataset classes may not have these attributes
        resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets = resampled_set.y_array
        try:  # Depending on the dataset these are responsible for the X features
            resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
        except:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError as e:
        try:
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        except:
            resampled_set_indices = np.concatenate(resampled_set_indices)
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
            print(f'Error: {e}')
    try:
        resampled_set.df = resampled_set.df.iloc[resampled_set_indices]
    except AttributeError:
        pass

    try:
        resampled_set.data = resampled_set.data[resampled_set_indices]
    except AttributeError:
        pass

    try:  # Depending on the dataset these are responsible for the X features
        resampled_set.filename_array = resampled_set.filename_array[resampled_set_indices]
    except:
        pass

    try:
        resampled_set.metadata_array = resampled_set.metadata_array[resampled_set_indices]
    except:
        pass
    
    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]
        
    return resampled_set


def imshow(img, mean=0.5, std=0.5):
    """
    Visualize data batches
    """
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
def plot_data_batch(dataset, mean=0.0, std=1.0, nrow=8, title=None,
                    args=None, save=False, save_id=None, ftype='png'):
    """
    Visualize data batches
    """
    try:
        img = make_grid(dataset, nrow=nrow)
    except Exception as e:
        raise e
        print(f'Nothing to plot!')
        return
    img = img * std + mean  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if save:
        try:
            fpath = join(args.image_path,
                         f'{save_id}-{args.experiment_name}.{ftype}')
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        except Exception as e:
            fpath = f'{save_id}-{args.experiment_name}.{ftype}'
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
    if args.display_image:
        plt.show()
    plt.close()
    
    
def get_datasets_by_spurious(initial_dataset, dataset_size, seed=42):
    np.random.seed(seed)
    targets = initial_dataset.targets_all['target']
    spurious = initial_dataset.targets_all['spurious']
    datasets = []
    
    class_size = int(np.round(dataset_size / len(np.unique(targets))))
    for s in np.unique(spurious):
        s_ix = np.where(spurious == s)[0]
        try:
            assert dataset_size < len(s_ix)
        except:
            raise AssertionError 
            
        y_indices = [np.where(targets[s_ix] == y)[0] 
                     for y in np.unique(targets[s_ix])]
        sample_indices = []
        
        for ix, y_ix in enumerate(y_indices):
            sample_ix = np.random.choice(s_ix[y_ix],
                                         size=class_size,
                                         replace=False,
                                         p=None)
            sample_indices.append(sample_ix)
                     
        sample_indices = np.concatenate(sample_indices)
        np.random.shuffle(sample_indices) 
        dataset = get_resampled_set(dataset=initial_dataset, 
                                    resampled_set_indices=sample_indices, 
                                    copy_dataset=True)
        datasets.append(dataset)
    return datasets


def get_dataset_by_spurious(initial_dataset, dataset_size, seed=42):
    np.random.seed(seed)
    targets = initial_dataset.targets_all['target']
    spurious = initial_dataset.targets_all['spurious']
    
    class_size = int(np.round(dataset_size / len(np.unique(targets))))
    all_sample_indices = []
    for s in np.unique(spurious):
        s_ix = np.where(spurious == s)[0]
        try:
            assert dataset_size < len(s_ix)
        except:
            raise AssertionError 
            
        y_indices = [np.where(targets[s_ix] == y)[0] 
                     for y in np.unique(targets[s_ix])]
        sample_indices = []
        
        for ix, y_ix in enumerate(y_indices):
            sample_ix = np.random.choice(s_ix[y_ix],
                                         size=class_size,
                                         replace=False,
                                         p=None)
            sample_indices.append(sample_ix)
            
        sample_indices = np.concatenate(sample_indices)
        all_sample_indices.append(sample_indices)
                     
    all_sample_indices = np.concatenate(all_sample_indices)
    np.random.shuffle(all_sample_indices) 
    dataset = get_resampled_set(dataset=initial_dataset, 
                                resampled_set_indices=all_sample_indices, 
                                copy_dataset=True)
    return dataset


def get_resampled_loader_by_predictions(predictions, initial_dataset, 
                                        resample, seed, args,
                                        batch_size=None):
    dataset = resample_dataset_by_predictions(predictions,
                                              initial_dataset,
                                              resample,
                                              args.group_size, 
                                              seed)
    batch_size = batch_size if batch_size is not None else args.bs_trn
    if args.dataset == 'fmow' and args.replicate in [6, 8, 9, 128] and len(dataset) % batch_size == 1:
        dataloader = DataLoader(dataset, shuffle=True,
                                batch_size=batch_size,
                                num_workers=args.num_workers,
                                drop_last=True)
        args.unlucky_leftover = True
    else:
        dataloader = DataLoader(dataset, shuffle=True,
                                batch_size=batch_size,
                                num_workers=args.num_workers)
        args.unlucky_leftover = False
    return dataloader


def resample_indices_by_class(indices, targets, resample, seed=42,
                              count_resampled_indices=False):
    np.random.seed(seed)
    _targets = targets[indices]
    resampled_indices = []
    indices_by_target = []
    group_sizes = []
    for t in np.unique(_targets):
        _indices = np.where(_targets == t)[0]
        indices_by_target.append(_indices)
        group_sizes.append(len(_indices))
        
    if resample == 'upsample':
        resample_size_ix = np.argmax(group_sizes)
        resample_size = group_sizes[resample_size_ix]
        for ix, _indices in enumerate(indices_by_target):
            if ix == resample_size_ix:
                resampled_indices.append(_indices)
            else:
                _indices = np.random.choice(_indices,
                                            size=resample_size,
                                            replace=True)
                resampled_indices.append(_indices)
    elif resample == 'subsample':
        resample_size_ix = np.argmin(group_sizes)
        resample_size = group_sizes[resample_size_ix]
        for ix, _indices in enumerate(indices_by_target):
            if ix == resample_size_ix:
                resampled_indices.append(_indices)
            else:
                _indices = np.random.choice(_indices,
                                            size=resample_size,
                                            replace=False)
                resampled_indices.append(_indices)
    else:
        resampled_indices = indices_by_target
        
    resampled_indices = np.concatenate(resampled_indices)
    if count_resampled_indices:
        print(np.unique(targets[indices[resampled_indices]], 
                        return_counts=True))
    return indices[resampled_indices]
    
"""
Group and class balancing
"""
def get_sample_indices_by_class_by_pred(class_labels,
                                        predictions,
                                        num_classes):
    """
    same thing as above
    """
    all_classes = range(num_classes)
    indices_by_class_by_pred = []
    
    for c in all_classes:
        indices_by_class = []
        for p in all_classes:
            indices = np.where(np.logical_and(class_labels == c,
                                              predictions == p))[0]
            indices_by_class.append(indices)
        indices_by_class_by_pred.append(indices_by_class)
    return indices_by_class_by_pred


def balance_groups(indices_by_group, resample='upsample', seed=42):
    np.random.seed(seed)
    resampled_indices = []
    sampling_indices = [i for i in indices_by_group if len(i) > 0]
    sizes = [len(i) for i in sampling_indices]
    
    if resample == 'upsample':
        ix_size = np.argmax(sizes)
        resample = True
    else:
        ix_size = np.argmin(sizes)
        resample = False
        
    for ix, indices in enumerate(sampling_indices):
        resample_size = sizes[ix_size]
        if ix == ix_size:
            resampled_indices.append(indices)
        else:
            resampled_indices.append(np.random.choice(indices, 
                                                      size=resample_size,
                                                      replace=resample))
    return np.concatenate(resampled_indices)

def balance_preds_by_class(indices_by_class_by_pred, seed):
    all_resampled_indices = []
    for class_indices in indices_by_class_by_pred:
        resampled_class_indices = balance_groups(class_indices,
                                                 resample='upsample',
                                                 seed=seed)
        all_resampled_indices.append(resampled_class_indices)
    all_resampled_indices = np.concatenate(all_resampled_indices)
    return all_resampled_indices


def balance_dataset_class_group(predictions, initial_dataset, seed):
    np.random.seed(seed)
    
    targets = initial_dataset.targets_all['target']
    try:
        predictions = predictions.numpy()
    except:
        pass
    
    indices_by_class = []
    for t in np.unique(targets):
        indices_by_class.append(np.where(targets == t)[0])
    balancing_indices = balance_groups(indices_by_class, seed=seed)
    _targets = targets[balancing_indices]
    _predictions =predictions[balancing_indices]
    
    indices_by_class_by_pred = get_sample_indices_by_class_by_pred(_targets, _predictions,
                                                                   num_classes=len(np.unique(targets)))
    
    _balancing_indices = balance_preds_by_class(indices_by_class_by_pred, seed)
    
    balancing_indices = balancing_indices[_balancing_indices]
    
    np.random.shuffle(balancing_indices)
    dataset_p = get_resampled_set(dataset=initial_dataset, 
                                  resampled_set_indices=balancing_indices, 
                                  copy_dataset=True)
    return dataset_p


def balance_dataset_group_class(predictions, initial_dataset, seed):
    np.random.seed(seed)
    
    targets = initial_dataset.targets_all['target']
    try:
        predictions = predictions.numpy()
    except:
        pass
    
    indices_by_class_by_pred = get_sample_indices_by_class_by_pred(targets, predictions,
                                                                   num_classes=len(np.unique(targets)))
    
    balancing_indices = balance_preds_by_class(indices_by_class_by_pred, seed)
    
    _targets = targets[balancing_indices]
    _predictions =predictions[balancing_indices]
    
    indices_by_class = []
    for t in np.unique(_targets):
        indices_by_class.append(np.where(_targets == t)[0])
        
    _balancing_indices = balance_groups(indices_by_class, seed)
    
    balancing_indices = balancing_indices[_balancing_indices]
    
    np.random.shuffle(balancing_indices)
    dataset_p = get_resampled_set(dataset=initial_dataset, 
                                  resampled_set_indices=balancing_indices, 
                                  copy_dataset=True)
    return dataset_p


def get_resampled_loader_by_groups(predictions, initial_dataset, 
                                   resample, seed, args,
                                   batch_size):
    np.random.seed(args.seed)
    if args.dataset == 'civilcomments':
        group_indices = initial_dataset.targets_all['spurious']
    else:
        group_indices = initial_dataset.targets_all['group_idx']
    all_group_indices = [np.where(group_indices == g)[0]
                         for g in np.unique(group_indices)]
    
    sample_indices = []
    if resample == 'subsample':
        sample_ix = np.argmin([len(i) for i in all_group_indices])
        sample_size = len(all_group_indices[sample_ix])
        
        for gix, g_indices in enumerate(all_group_indices):
            if gix != sample_ix:
                sample_indices.append(
                    np.random.choice(g_indices,
                                     size=sample_size,
                                     replace=False,
                                     p=None))
            else:
                sample_indices.append(g_indices)
    elif resample == 'upsample':
        sample_ix = np.argmax([len(i) for i in all_group_indices])
        sample_size = len(all_group_indices[sample_ix])
        for gix, g_indices in enumerate(all_group_indices):
            sample_indices.append(g_indices)
            if len(g_indices) < sample_size:
                sample_indices.append(
                    np.random.choice(g_indices,
                                     size=sample_size - len(g_indices),
                                     replace=True,
                                     p=None))
    elif resample == 'group_size':
        group_size = args.group_size
        assert group_size is not None
        for gix, g_indices in enumerate(all_group_indices):
            replace = True if len(g_indices) < group_size else False
            sample_indices.append(
                np.random.choice(g_indices,
                                 size=group_size,
                                 replace=replace,
                                 p=None)
            )
    sample_indices = np.concatenate(sample_indices)
    np.random.shuffle(sample_indices)
    dataset = get_resampled_set(dataset=initial_dataset, 
                                resampled_set_indices=sample_indices, 
                                copy_dataset=True)
    
    batch_size = batch_size if batch_size is not None else args.bs_trn
    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=batch_size,
                            num_workers=args.num_workers)
    return dataloader
        



def resample_dataset_by_predictions(predictions, initial_dataset, 
                                    resample, group_size=None,
                                    seed=42):
    np.random.seed(seed)
    targets = initial_dataset.targets_all['target']
    all_sample_indices = []
    if resample == 'incorrect':
        incorrect_ix = np.where(targets != predictions)[0]
        all_sample_indices.append(incorrect_ix)
    else:
        for p in np.unique(predictions):
            p_ix = np.where(predictions == p)[0]
            y_indices = [np.where(targets[p_ix] == y)[0]
                         for y in np.unique(targets[p_ix])]
            if resample == 'upsample':
                max_ix = np.argmax([len(y_ix) for y_ix in y_indices])
                max_len = len(y_indices[max_ix])
                sample_indices = [p_ix[y_ix] for y_ix in copy.deepcopy(y_indices)]
                for ix, y_ix in enumerate(y_indices):
                    if ix != max_ix:
                        sample_indices.append(
                            np.random.choice(p_ix[y_ix],
                                             size=max_len - len(y_ix), 
                                             replace=True,
                                             p=None))
                p_ix = np.concatenate(sample_indices)
                np.random.shuffle(p_ix) 
            elif resample == 'subsample':
                min_ix = np.argmin([len(y_ix) for y_ix in y_indices])
                sample_indices = [p_ix[y_indices[min_ix]]]
                for ix, y_ix in enumerate(y_indices):
                    if ix != min_ix:
                        sample_indices.append(
                            np.random.choice(p_ix[y_ix],
                                             size=len(y_indices[min_ix]),
                                             replace=False,
                                             p=None))
                p_ix = np.concatenate(sample_indices)

            elif resample == 'group_size':
                assert group_size is not None
                sample_indices = []
                for ix, y_ix in enumerate(y_indices):
                    replace = True if len(p_ix[y_ix]) < group_size else False
                    sample_indices.append(
                        np.random.choice(p_ix[y_ix],
                                         size=group_size,
                                         replace=replace,
                                         p=None))
                p_ix = np.concatenate(sample_indices)

            all_sample_indices.append(p_ix)
        
    p_ix = np.concatenate(all_sample_indices)
    np.random.shuffle(p_ix)
    dataset_p = get_resampled_set(dataset=initial_dataset, 
                                  resampled_set_indices=p_ix, 
                                  copy_dataset=True)
    ratio = len(p_ix) / len(targets) * 100
    group_ix, group_counts = np.unique(dataset_p.targets_all['group_idx'], return_counts=True)
    return dataset_p


def get_group_labels_from_class_pred(predictions, class_labels, num_classes):
    # Ex.) [0, 0] -> 0; [0, 1] -> 1; [1, 0] -> 2; [1, 1] -> 3
    group_label_func = lambda ix: predictions[ix] * num_classes + class_labels[ix]
    new_labels = np.array(group_label_func(np.arange(len(predictions))))
    return new_labels


def get_class_label_from_group_pred(predictions, num_classes=None, mapping=None):
    # Example: mapping = [0, 1, 0, 1]
    if mapping is None:
        raise NotImplementedError
        
    else:
        get_class_label = np.vectorize(lambda x: mapping[x])
        class_labels = get_class_label(predictions)
    return class_labels


def get_dataloader_with_pred_group_labels(dataloader, pred_group_labels,
                                          **kwargs):
    dataset = copy.deepcopy(dataloader.dataset)
    dataset.targets = torch.tensor(pred_group_labels)
    dataset.targets_all['pred_group_target'] = pred_group_labels
    
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader


def load_data_with_pred_group_labels(dataloaders, predictions, args):
    dataloaders_with_pred_group_labels = []
    for ix, dataloader in enumerate(dataloaders):
        class_labels = dataloader.dataset.targets_all['target']
        pred_group_labels = get_group_labels_from_class_pred(predictions[ix],
                                                             class_labels, 
                                                             args.num_classes)
        if ix == 0:
            shuffle = True; batch_size = args.bs_trn
        else:
            shuffle = False; batch_size = args.bs_val
        dataloader = get_dataloader_with_pred_group_labels(dataloader, 
                                                           pred_group_labels, 
                                                           shuffle=shuffle, 
                                                           batch_size=batch_size, 
                                                           num_workers=args.num_workers)
        dataloaders_with_pred_group_labels.append(dataloader)
    return dataloaders_with_pred_group_labels


# Define alternate resampling method
def resample_dataset_by_pred_by_class(predictions, initial_dataset, 
                                      resample, group_size=None,
                                      seed=42):
    np.random.seed(seed)
    targets = initial_dataset.targets_all['target']
    all_sample_indices = []
    for p in np.unique(targets):
        p_ix = np.where(targets == p)[0]
        y_indices = [np.where(predictions[p_ix] == y)[0]
                     for y in np.unique(predictions[p_ix])]
        if resample == 'upsample':
            max_ix = np.argmax([len(y_ix) for y_ix in y_indices])
            max_len = len(y_indices[max_ix])
            sample_indices = [p_ix[y_ix] for y_ix in copy.deepcopy(y_indices)]
            for ix, y_ix in enumerate(y_indices):
                if ix != max_ix:
                    sample_indices.append(
                        np.random.choice(p_ix[y_ix],
                                         size=max_len - len(y_ix), 
                                         replace=True,
                                         p=None))
            p_ix = np.concatenate(sample_indices)
            np.random.shuffle(p_ix) 
        elif resample == 'subsample':
            min_ix = np.argmin([len(y_ix) for y_ix in y_indices])
            sample_indices = [p_ix[y_indices[min_ix]]]
            for ix, y_ix in enumerate(y_indices):
                if ix != min_ix:
                    sample_indices.append(
                        np.random.choice(p_ix[y_ix],
                                         size=len(y_indices[min_ix]),
                                         replace=False,
                                         p=None))
            p_ix = np.concatenate(sample_indices)
      
        elif resample == 'group_size':
            assert group_size is not None
            sample_indices = []
            for ix, y_ix in enumerate(y_indices):
                replace = True if len(p_ix[y_ix]) < group_size else False
                sample_indices.append(
                    np.random.choice(p_ix[y_ix],
                                     size=group_size,
                                     replace=replace,
                                     p=None))
            p_ix = np.concatenate(sample_indices)
            
        all_sample_indices.append(p_ix)
        
    p_ix = np.concatenate(all_sample_indices)
    np.random.shuffle(p_ix)
    dataset_p = get_resampled_set(dataset=initial_dataset, 
                                  resampled_set_indices=p_ix, 
                                  copy_dataset=True)
    ratio = len(p_ix) / len(targets) * 100 
    group_ix, group_counts = np.unique(dataset_p.targets_all['group_idx'], return_counts=True)
    return dataset_p


def get_resampled_loader_by_pred_by_class(predictions, initial_dataset, 
                                          resample, seed, args,
                                          batch_size=None):
    dataset = resample_dataset_by_pred_by_class(predictions, 
                                                initial_dataset,
                                                resample, 
                                                args.group_size,
                                                seed)
    batch_size = batch_size if batch_size is not None else args.bs_trn
    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=batch_size,
                            num_workers=args.num_workers)
    return dataloader





# Class balancing
def get_indices_to_balance_classes(dataloader, resample='subsample',
                                   seed=42):
    np.random.seed(seed)
    targets = dataloader.dataset.targets_all['target']
    all_sample_indices = []
    for t in np.unique(targets):
        t_ix = np.where(targets == t)[0]
        all_sample_indices.append(t_ix)
        
    if resample == 'subsample':
        sample_ix = np.argmin([len(i) for i in all_sample_indices])
        replace = False
    else:
        sample_ix = np.argmax([len(i) for i in all_sample_indices])
        replace = True
    sample_size = len(all_sample_indices[sample_ix])
    
    for ix, s_ix in enumerate(all_sample_indices):
        if ix != sample_ix:  # Keep the original
            all_sample_indices[ix] = np.random.choice(s_ix,
                                                      size=sample_size,
                                                      replace=replace,
                                                      p=None)
    all_sample_indices = np.concatenate(all_sample_indices)
    np.random.shuffle(all_sample_indices)
    return all_sample_indices


def get_balanced_class_dataset(dataloader, resample='subsample', seed=42):
    resampled_indices = get_indices_to_balance_classes(
        dataloader, resample, seed)
    dataset = get_resampled_set(dataset=dataloader.dataset,
                                resampled_set_indices=resampled_indices,
                                copy_dataset=True)
    return dataset


def get_balanced_class_dataloader(dataloader, resample='subsample',
                                  seed=42, args=None, batch_size=None):
    dataset = get_balanced_class_dataset(
        dataloader, resample, seed)
    batch_size = batch_size if batch_size is not None else args.bs_trn
    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=batch_size,
                            num_workers=args.num_workers)
    return dataloader
    

