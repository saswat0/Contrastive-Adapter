"""
BREEDS Datasets
See https://github.com/MadryLab/BREEDS-Benchmarks for reference
"""

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from datasets import train_val_split
from utils.visualize import plot_data_batch

from datasets.data.BREEDS.breeds_helpers import make_entity13, make_entity30, make_living17, make_nonliving26


class BREEDSDataset(Dataset):
    def __init__(self, root_dir, target_name, confounder_names, split,
                 dataset_name='entity13',  
                 subclasses=['source'],
                 imagenet_dir='./datasets/data/imagenet',
                 transform=None):
        self.root_dir = root_dir
        self.target_name = target_name  # superclass
        self.confounder_names = confounder_names  # subclass
        self.dataset_name = dataset_name  # 'entity13', 'entity30', 'living17', 'nonliving26'
        self.subclasses = subclasses
        self.transform = transform
        
        # For adding target group to training data
        self.target_group_ratio = 0
        self.split = split
        
        # Hold-out part of official train split for val set
        imagenet_split = 'val' if split == 'test' else 'train' 
        breeds_datasets = ['entity13', 'entity30', 'living17', 'nonliving26']
        subclass_setups = [['source'], ['target'], ['source', 'target']]
        
        assert self.dataset_name in breeds_datasets, f"Please use a BREEDS dataset name from {breeds_datasets}"
        # assert self.subclasses in subclass_setups, f"Please specify a subclass setup from {subclass_setups}"
        
        self.source_dataset = torchvision.datasets.ImageNet(imagenet_dir, 
                                                            split=imagenet_split,
                                                            transform=transform)
        
        superclasses, subclass_split, label_map = init_breeds_class_info(root_dir, 
                                                                         dataset_name)
        subtarget_set = self.init_subtarget_maps(subclass_split, label_map)
        self.subclass_split = subclass_split
        
        # Initialize data
        self.targets, self.sub_targets, mask = self.init_data(subtarget_set,
                                                              self.source_dataset)
        # Filter images to BREEDS classes
        self.source_dataset.samples = np.array(self.source_dataset.samples)[mask]
        
        self.mask = mask
        
        if split == 'train':
            indices = train_val_split(self.targets, 0.2, seed=42)
            self.targets = self.targets[indices[0]]
            self.sub_targets = self.sub_targets[indices[0]]
            self.source_dataset.samples = self.source_dataset.samples[indices[0]]
            self.mask = mask[indices[0]]
            
        elif split == 'val':
            indices = train_val_split(self.targets, 0.2, seed=42)
            self.targets = self.targets[indices[-1]]
            self.sub_targets = self.sub_targets[indices[-1]]
            self.source_dataset.samples = self.source_dataset.samples[indices[-1]]
            self.mask = mask[indices[-1]]
            
        self.targets_all = {'target': self.targets.numpy(),
                            'spurious': self.sub_targets.numpy(),
                            'sub_target': self.sub_targets.numpy(),
                            'group_idx': self.sub_targets.numpy()}
        
        # For WILDS evaluation
        self.y_array = self.targets
        self.metadata_array = self.sub_targets
        self.categories = list(label_map.values())
        self.text_descriptions = [f'a picture of an {c}' if c[0] in ['a', 'e', 'i', 'o', 'u']
                                  else f'a picture of a {c}' for c in self.categories]
        
        
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Transform should be applied here too        
        x, _ = self.source_dataset.__getitem__(idx)
        y = self.targets[idx]
        metadata = self.sub_targets[idx]
        return x, y, metadata
        
    def init_data(self, subtarget_set, dataset):
        if self.target_group_ratio > 0:
            print(f'-> target_group_ratio: {self.target_group_ratio}')
            source_subclasses, target_subclasses = self.subclass_split
            _mask_source = [np.where(self.source_dataset.targets == t)[0] 
                            for t in np.array(source_subclasses).flatten()]

            _mask_target = [np.where(self.source_dataset.targets == t)[0] 
                            for t in np.array(target_subclasses).flatten()]

            class_size = len(_mask_source[0])
            _mask_source = np.concatenate(_mask_source)

            # Subsample the targets
            np.random.choice(42)
            sample_size = ((self.target_group_ratio * class_size) / 
                           (1 - self.target_group_ratio))
            sample_size = int(np.round(sample_size))
            print(f'  -> target group ratio: {sample_size} / ({sample_size} + {class_size}) = {sample_size / (sample_size + class_size)}')

            for ix, indices in enumerate(_mask_target):
                _mask_target[ix] = np.random.choice(indices,
                                                    size=sample_size,
                                                    replace=False)
            _mask_target = np.concatenate(_mask_target)

            mask = np.concatenate((_mask_source, _mask_target))
        
        else:
            mask = np.where(np.logical_or.reduce(
                [dataset.targets == t for t in subtarget_set.flatten()]
            ))[0]
        sub_targets = torch.tensor(dataset.targets)[mask]
        targets = torch.tensor([self.subtarget_to_target_map[t.item()] for t in sub_targets])
        return targets, sub_targets, mask
        
    def init_subtarget_maps(self, subclass_split, label_map):
        source_subclasses, target_subclasses = subclass_split
        if self.subclasses == ['source']:
            subtargets = np.array(source_subclasses)
        elif self.subclasses == ['target']:
            subtargets = np.array(target_subclasses)
        else:
            subtargets = np.hstack([source_subclasses, 
                                    target_subclasses])
            
        self.subtarget_to_target_map = {}
        self.subtarget_to_subclass_map = {}
        for class_id in range(len(subtargets)):
            for subclass_id in subtargets[class_id]:
                self.subtarget_to_target_map[subclass_id] = class_id
                self.subtarget_to_subclass_map[subclass_id] = self.source_dataset.classes[subclass_id][0]
                
        self.subtarget_to_class_map = {k: label_map[v % len(label_map)] for k, v in 
                                       self.subtarget_to_target_map.items()}
        try:
            if float(self.subclasses[-1]) > 0 and self.split != 'test':
                self.target_group_ratio = float(self.subclasses[-1])
        except:
            pass
        
        return subtargets
    
    def eval(self, y_pred, y_true, metadata):
        # WILDS style evaluation
        results = {'acc_avg': 0.,
                   'acc_wg': 101.}
        
        all_correct = np.array(y_pred == y_true)
        results['acc_avg'] = all_correct.sum() / len(y_true)
        
        superclass_accs = {c: {'correct': 0, 'total': 0, 'subclass': {}} 
                           for c in self.categories}
        
        max_superclass_len = 0
        max_subclass_len = 0
        
        for ix in np.unique(metadata.numpy()):
            subclass_idx = np.where(metadata.numpy() == ix)[0]
            subclass     = self.subtarget_to_subclass_map[ix]
            superclass   = self.subtarget_to_class_map[ix]
            if len(superclass) > max_superclass_len:
                max_superclass_len = len(superclass)
                
            if len(subclass) > max_subclass_len:
                max_subclass_len = len(subclass)
            
            subclass_correct = all_correct[subclass_idx].sum()
            subclass_total   = len(subclass_idx)
            
            superclass_accs[superclass]['correct'] += subclass_correct
            superclass_accs[superclass]['total']   += subclass_total
            
            superclass_accs[superclass]['subclass'][subclass] = {
                'correct': subclass_correct,
                'total':   subclass_total
            }
            try:
                results[f'acc_subclass={ix}'] = subclass_correct / subclass_total
            except:
                results[f'acc_subclass={ix}'] = 0
                
        results_str = f"Average acc: {results['acc_avg'] * 100:.3f}\n"
        
        for superclass in superclass_accs:
            try:
                acc = (superclass_accs[superclass]['correct'] / 
                       superclass_accs[superclass]['total'])
            except:
                acc = 0.
            total = superclass_accs[superclass]['total']
            results_str += f'{superclass:{max_superclass_len}s} acc: {acc * 100:.3f} (n = {total:4d})\n'
            for subclass, metrics in superclass_accs[superclass]['subclass'].items():
                try:
                    acc_ = metrics['correct'] / metrics['total']
                except:
                    acc_ = 0.
                total_ = metrics['total']
                results_str += f'    - {subclass:{max_subclass_len}s} acc: {acc_ * 100:.3f} (n = {total_:3d})\n'
                results[f'acc_class={superclass}-subclass={subclass.replace(" ", "_")}'] = acc_
                if acc_ < results['acc_wg']:
                    results['acc_wg'] = acc_
                    
            results[f'acc_class={superclass}'] = acc
        results_str += f"Worst-group acc: {results['acc_wg'] * 100:.3f}"
                
        return results, results_str
    
    
def load_breeds(args, train_shuffle=True, transform=None):
    dataloaders = []
    for split in ['train', 'val', 'test']:
        batch_size = args.bs_trn if split == 'train' else args.bs_val
        shuffle = True if (train_shuffle and 'split' == 'train') else False
        dataset = BREEDSDataset(args.root_dir, 
                                target_name=args.target_name,
                                confounder_names=args.confounder_names, 
                                split=split,
                                dataset_name=args.breeds_dataset_name,
                                subclasses=args.breeds_subclasses,
                                transform=transform)
        dataloader = DataLoader(dataset, 
                                shuffle=shuffle, 
                                batch_size=batch_size,
                                num_workers=args.num_workers)
        print(f'Split: {split}, dataset size: {len(dataset)}, dataloader size: {len(dataloader)}')
        dataloaders.append(dataloader)
    args.text_descriptions = dataloaders[0].dataset.text_descriptions
    args.train_classes = dataloaders[0].dataset.categories
    args.num_classes = len(args.train_classes)
    return dataloaders


def load_dataloaders(args, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_breeds(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    # Filter for selected datapoints (in case we use SubsetRandomSampler)
    try:
        subset_indices = dataloader.sampler.indices
        targets = dataloader.dataset.targets_all[target_type][subset_indices]
        subset = True
    except AttributeError:
        targets = dataloader.dataset.targets_all[target_type]
        subset = False
    all_data_indices = []
    for class_ in np.unique(targets):
        class_indices = np.where(targets == class_)[0]
        all_data_indices.extend(class_indices[:num_datapoints])
        
    nrow = num_datapoints if num_datapoints < 8 else 8
    
    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] 
                     for ix in all_data_indices],
                    mean=np.mean((0.48145466, 0.4578275, 0.40821073)),
                    std=np.mean((0.26862954, 0.26130258, 0.27577711)),
                    nrow=nrow, title=title,
                    args=args, save=save, save_id=save_id, ftype=ftype)
    return NotImplementedError
        
        
def init_breeds_class_info(root_dir, dataset_name):
    if dataset_name == 'entity13':
        superclasses, subclass_split, label_map = make_entity13(root_dir, 
                                                                split='rand')
        label_map[2]  = 'reptile'
        label_map[4]  = 'mammal'
        label_map[5]  = 'accessory'
        label_map[8]  = 'piece of furniture'
        label_map[10] = 'structure'
        label_map[11] = 'vehicle'
        label_map[12] = 'fruit or vegetable'

    elif dataset_name == 'entity30':
        superclasses, subclass_split, label_map = make_entity30(root_dir, 
                                                                split='rand')
        label_map[0]  = 'snake'
        label_map[1]  = 'land bird'
        label_map[2]  = 'lizard'
        label_map[3]  = 'arachnid or spider'
        label_map[8]  = 'hoofed mammal'
        label_map[12] = 'building'
        label_map[14] = 'pair of footwear or legwear'
        label_map[15] = 'garment'
        label_map[16] = 'headgear'
        label_map[17] = 'home appliance'
        label_map[19] = 'measuring instrument'
        label_map[20] = 'motor vehicle'
        label_map[21] = 'musical instrument'
        label_map[26] = 'vessel or boat'
        label_map[27] = 'food'
        label_map[28] = 'vegetables'

    elif dataset_name == 'living17':
        superclasses, subclass_split, label_map = make_living17(root_dir, 
                                                                split='rand')
        label_map[3]  = 'snake'
        label_map[8]  = 'dog'
        label_map[11] = 'cat'

    elif dataset_name == 'nonliving26':
        superclasses, subclass_split, label_map = make_nonliving26(root_dir, 
                                                                   split='rand')
        label_map[3]  = 'armor'
        label_map[5]  = 'bus'
        label_map[6]  = 'car'
        label_map[9]  = 'computer'
        label_map[10] = 'home or dwelling'
        label_map[11] = 'fence'
        label_map[12] = 'hat'
        label_map[14] = 'store or shop'
        label_map[16] = 'percussion instrument'
        label_map[22] = 'timepiece'
        label_map[23] = 'truck'
        label_map[24] = 'wind instrument'

    return superclasses, subclass_split, label_map
          
            

