"""
CivilComments-WILDS Dataset

Setup from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/civilcomments_dataset.py

Relies on the WILDS module, which can be installed with `pip install wilds`
Alternatively, pull the main branch from GitHub and include the `wilds/wilds` directory in this project directory.  
"""

import os
from os.path import join
from copy import deepcopy

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class CivilComments(Dataset):
    """
    CivilComments dataset
    """
    
    def __init__(self, root_dir, target_name, confounder_names,
                 args=None, transform=None):
        
        self.root_dir = join(root_dir, 'civilcomments_v1.0')
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.args = args
        self.transform = transform
        
        # Main dataset courtesy of WILDS
        self.wilds_dataset = get_dataset(dataset='civilcomments',
                                         download=False,
                                         root_dir=root_dir)
        
        self.categories = ['non_toxic', 'toxic']
        self.prompt_templates = ['this is a toxic comment']

    
def initialize_targets(dataloader):
    get_spurious = lambda x: '_'.join(x)
    spurious = np.apply_along_axis(
        get_spurious, 1, dataloader.dataset.metadata_array[:, :-2].numpy().astype(str))
    spurious_to_idx = {cat: i for i, cat in enumerate(np.unique(spurious))}
    targets_t = dataloader.dataset.y_array.numpy()
    targets_s = np.asarray([spurious_to_idx[y] for y in list(spurious)])

    targets = dataloader.dataset.y_array
    targets_all = {'target': targets_t, 'spurious': targets_s}
    return targets, targets_all
        

def load_civilcomments(args, train_shuffle=True, transform=None):
    dataset = CivilComments(args.root_dir,
                   target_name='toxic',
                   confounder_names=['identities'],
                   args=args, transform=None) 
    if args.dataset == 'civilcomments_line':
        all_data_text_array = dataset.wilds_dataset._text_array 
        all_data_text_array = [str(t.replace('\n', ' ')).replace("\'", "'") for t in all_data_text_array]
        dataset.wilds_dataset._text_array = all_data_text_array 
    
    dataloaders = []
    for ix, split in enumerate(['train', 'val', 'test']):
        batch_size = args.bs_trn if ix == 0 else args.bs_val
        shuffle = True if (train_shuffle and 'split' == 'train') else False
        _dataset = dataset.wilds_dataset.get_subset(split, transform=transform)
        dataloader = DataLoader(_dataset, shuffle=shuffle, sampler=None,
                                collate_fn=_dataset.collate, batch_size=batch_size)
        targets, targets_all = initialize_targets(dataloader)
        dataloader.dataset.targets = targets
        dataloader.dataset.targets_all = targets_all
        # CLIP Prompts
        dataloader.dataset.classes = dataset.categories
        dataloader.dataset.prompts = dataset.prompt_templates
        # WILDS Eval
        dataloader.dataset.eval = dataset.wilds_dataset.eval
        dataloader.wilds_dataset = dataset.wilds_dataset
        dataloaders.append(dataloader)
        
    return dataloaders


def load_dataloaders(args, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_civilcomments(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    """
    Does not currently apply to NLP datasets
    """
    return 
