"""
Amazon-WILDS Dataset

Setup from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/civilcomments_dataset.py

Relies on the WILDS module, which can be installed with `pip install wilds`
Alternatively, pull the main branch from GitHub and include the `wilds/wilds` directory in this project directory.  
"""

import os
from os.path import join
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class Amazon(Dataset):
    """
    Amazon dataset
    """
    
    def __init__(self, root_dir, target_name, confounder_names,
                 args=None, transform=None, 
                 split_scheme='category_subpopulation'):
        
        self.root_dir = join(root_dir, 'amazon_v2.1')
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.args = args
        self.transform = transform
        
        # Main dataset courtesy of WILDS
        self.wilds_dataset = get_dataset(dataset='amazon',
                                         download=False,
                                         root_dir=root_dir,
                                         split_scheme=split_scheme)
        
        self.categories = ['negative', 'positive']
        # Only one prompt because binary classification
        # - Do zero-shot classification via NLI as in Yin et al. (2019) 
        #   https://arxiv.org/abs/1909.00161
        self.prompt_templates = ['this is a {} review']
        
        
def get_binary_targets(y_array):
    """
    Convert 0, 1 to 0; 3, 4 to 1, keep 2
    """
    y_array[y_array < 2] = 0
    y_array[y_array > 2] = 1
    return y_array

    
def initialize_targets(dataset, split_scheme):
    """
    Gets rid of 3-star reviews. Changes 0, 1-star to negative reviews; 4, 5-star reviews to positive reviews
    """
    _targets = dataset.y_array 
    keep_indices = np.where(_targets != 2)[0]
    
    # wilds subset
    dataset.indices = dataset.indices[keep_indices]
    
    assert len(dataset.y_array) == len(keep_indices)
    
    targets = dataset.y_array
    
    targets_t = dataset.y_array.numpy()
    if split_scheme == 'subpopulation':
        targets_s = dataset.metadata_array[:, 2].numpy()
    elif split_scheme == 'user':
        targets_s = dataset.metadata_array[:, 0].numpy()
    else:
        raise NotImplementedError
    
    targets_all = {'target': targets_t, 'spurious': targets_s}
    return targets, targets_all
        

def load_amazon(args, train_shuffle=True, transform=None):
    confounder = args.dataset.split('_')[-1]  # subpopulation, user
    split_scheme = 'category_subpopulation' if confounder == 'subpopulation' else 'user'
    dataset = Amazon(args.root_dir,
                   target_name='negative',
                   confounder_names=[confounder],
                   args=args, transform=transform,
                     split_scheme=split_scheme)
    # Make reviews one line
    all_data_text_array = dataset.wilds_dataset._input_array
    all_data_text_array = [str(t.replace('\n', ' ')).replace("\'", "'") for t in all_data_text_array]
    dataset.wilds_dataset._input_array = all_data_text_array
    # Make review labels 0 or 1
    dataset.wilds_dataset._y_array = get_binary_targets(dataset.wilds_dataset._y_array)
    dataloaders = []
    for ix, split in enumerate(['train', 'val', 'test']):
        batch_size = args.bs_trn if ix == 0 else args.bs_val
        shuffle = True if (train_shuffle and 'split' == 'train') else False
        _dataset = dataset.wilds_dataset.get_subset(split, transform=transform)
        targets, targets_all = initialize_targets(_dataset, confounder)
        _dataset.targets = targets
        _dataset.targets_all = targets_all
        _dataset.metadata_array[:, 4] = targets
        # CLIP Prompts
        _dataset.classes = dataset.categories
        _dataset.dataset.prompts = dataset.prompt_templates
        # WILDS Eval
        _dataset.dataset.eval = dataset.wilds_dataset.eval
        _dataset.wilds_dataset = dataset.wilds_dataset
        dataloader = DataLoader(_dataset, shuffle=shuffle, sampler=None,
                                collate_fn=_dataset.collate, batch_size=batch_size)
        
        dataloaders.append(dataloader)
        
    return dataloaders


def load_dataloaders(args, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_amazon(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    """
    Does not currently apply to NLP datasets
    """
    return 
