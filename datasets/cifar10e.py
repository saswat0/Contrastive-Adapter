"""
CIFAR 10.001 and CIFAR 10.02 Datasets
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from os.path import join

from datasets import train_val_split


class CIFAR10(Dataset):
    def __init__(self, target_name, confounder_names, class_names, group_names,
                 data_a, data_b, targets_a, targets_b, split, transform):
        self.target_name = target_name  # superclass
        self.confounder_names = confounder_names  # subclass
        self.class_names = class_names
        self.group_names = group_names
        
        # Combine
        self.data = np.vstack((data_a, data_b))
        self.targets = np.concatenate((targets_a, targets_b))
        # whether from CIFAR-10 or CIFAR-10.1
        self.spurious = (np.arange(len(self.targets)) >= len(targets_a)).astype(int)
        
        print(f' -> group ratio: {self.spurious.sum()} / ({self.spurious.sum()} + {len(data_a)}) = {self.spurious.sum() / (self.spurious.sum() + len(data_a))}')
        
        # Get groups and group to class assignment
        self.groups = np.zeros(len(self.targets)).astype(int)
        self.subtarget_to_target_map = {}
        self.subtarget_to_class_map = {}
        self.subtarget_to_subclass_map = {}
        group_idx = 0
        for t in np.unique(self.targets):
            for s in np.unique(self.spurious):
                indices = np.where(np.logical_and(
                    self.targets == t, self.spurious == s
                ))[0]
                self.groups[indices] += group_idx
                self.subtarget_to_target_map[group_idx] = t
                self.subtarget_to_class_map[group_idx] = self.class_names[t]
                self.subtarget_to_subclass_map[group_idx] = self.group_names[s]
                group_idx += 1
        
        self.transform_to_pilimage = torchvision.transforms.ToPILImage()
        self.transform = transform
        
        if split == 'train':
            indices = train_val_split(self.targets, 0.2, seed=42)
            self.targets = self.targets[indices[0]]
            self.spurious = self.spurious[indices[0]]
            self.groups = self.groups[indices[0]]
            self.data = self.data[indices[0]]
            
        elif split == 'val':
            indices = train_val_split(self.targets, 0.2, seed=42)
            self.targets = self.targets[indices[-1]]
            self.spurious = self.spurious[indices[-1]]
            self.groups = self.groups[indices[-1]]
            self.data = self.data[indices[-1]]
            
        self.targets_all = {'target': self.targets,
                            'spurious': self.spurious, 
                            'group_idx': self.groups}
        
        self.targets = torch.from_numpy(self.targets).type(torch.long)
        self.spurious = torch.from_numpy(self.spurious).type(torch.long)
        self.groups = torch.from_numpy(self.groups).type(torch.long)
        
        # For WILDS evaluation
        self.y_array = self.targets
        self.metadata_array = self.groups
        
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform_to_pilimage(x)
            x = self.transform(x)
        y = self.targets[idx]
        z = self.metadata_array[idx]
        return x, y, z
    
    
    def __len__(self):
        return len(self.targets)
    
    
    def eval(self, y_pred, y_true, metadata):
        # WILDS style evaluation
        results = {'acc_avg': 0.,
                   'acc_wg': 101.}
        
        all_correct = np.array(y_pred == y_true)
        results['acc_avg'] = all_correct.sum() / len(y_true)
        
        superclass_accs = {c: {'correct': 0, 'total': 0, 'subclass': {}} 
                           for c in self.class_names}
        max_class_len = 0
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


def load_cifar10e1(args, train_shuffle=True, transform=None):
    """
    CIFAR-10.002: 
    - CIFAR-10 + CIFAR-10.1 where CIFAR-10.1 makes up 2% of training samples
    """
    group_names = ['CIFAR-10', 'CIFAR-10.1']
    # Load CIFAR-10.1 data and split up
    root_dir_1 = './datasets/data/cifar10.1/'
    data_cifar10_1 = np.load(join(root_dir_1, 'cifar10.1_v6_data.npy'))
    targets_cifar10_1 = np.load(join(root_dir_1, 'cifar10.1_v6_labels.npy'))
    indices_by_target_1 = [np.where(targets_cifar10_1 == t)[0]
                           for t in range(10)]
    np.random.seed(42)
    train_val_indices_1 = np.concatenate([
        np.random.choice(indices, size=102, replace=True) 
        for indices in indices_by_target_1
    ])
    test_indices_1 = np.setdiff1d(np.arange(len(targets_cifar10_1)), 
                                  train_val_indices_1)
    
    dataloaders = []
    
    for split in ['train', 'val', 'test']:
        batch_size = args.bs_trn if split == 'train' else args.bs_val
        train_split = False if split == 'test' else True
        shuffle = True if (train_shuffle and 'split' == 'train') else False
        
        dataset = torchvision.datasets.CIFAR10(root=args.root_dir,
                                               transform=transform,
                                               train=train_split)
        data_cifar10 = dataset.data
        targets_cifar10 = np.array(dataset.targets)
        
        cifar_10_1_indices = (test_indices_1 if split == 'test' 
                              else train_val_indices_1)
        _data_cifar10_1 = data_cifar10_1[cifar_10_1_indices]
        _targets_cifar10_1 = targets_cifar10_1[cifar_10_1_indices]
        
        dataset = CIFAR10(target_name=args.target_name,
                          confounder_names=args.confounder_names,
                          class_names=dataset.classes,
                          group_names=group_names,
                          data_a=data_cifar10,
                          data_b=_data_cifar10_1,
                          targets_a=targets_cifar10,
                          targets_b=_targets_cifar10_1,
                          split=split,
                          transform=transform)
        dataloader = DataLoader(dataset, 
                                shuffle=shuffle, 
                                batch_size=batch_size,
                                num_workers=args.num_workers)
        dataloaders.append(dataloader)
    return dataloaders


def load_cifar10e2(args, train_shuffle=True, transform=None):
    """
    CIFAR-10.02: 
    - CIFAR-10 + CIFAR-10.2 where CIFAR-10.2 makes up 10% of training samples
    """
    group_names = ['CIFAR-10', 'CIFAR-10.2']
    root_dir_1 = './datasets/data/cifar10.2/'
    train_data_1 = np.load(join(root_dir_1, 'cifar102_train.npz'))
    test_data_1 = np.load(join(root_dir_1, 'cifar102_test.npz'))
    
    train_targets_1 = train_data_1['labels']
    test_targets_1  = test_data_1['labels']
    train_images_1  = train_data_1['images']
    test_images_1   = test_data_1['images']
    
    train_indices_by_target_1 = [np.where(train_targets_1 == t)[0]
                                 for t in range(10)]
    np.random.seed(42) 
    train_val_indices_1 = np.concatenate([
        np.random.choice(indices, size=556, replace=True) 
        for indices in train_indices_by_target_1
    ])
    test_indices_1 = np.arange(len(test_targets_1))
    
    dataloaders = []
    
    for split in ['train', 'val', 'test']:
        batch_size = args.bs_trn if split == 'train' else args.bs_val
        train_split = False if split == 'test' else True
        shuffle = True if (train_shuffle and 'split' == 'train') else False
        
        dataset = torchvision.datasets.CIFAR10(root=args.root_dir,
                                               transform=transform,
                                               train=train_split)
        data_cifar10 = dataset.data
        targets_cifar10 = np.array(dataset.targets)
        
        cifar_10_1_indices = (test_indices_1 if split == 'test' 
                              else train_val_indices_1)
        _data_cifar10_1 = test_images_1 if split == 'test' else train_images_1
        _data_cifar10_1 = _data_cifar10_1[cifar_10_1_indices]
        _targets_cifar10_1 = test_targets_1 if split == 'test' else train_targets_1
        _targets_cifar10_1 =  _targets_cifar10_1[cifar_10_1_indices]
        
        dataset = CIFAR10(target_name=args.target_name,
                          confounder_names=args.confounder_names,
                          class_names=dataset.classes,
                          group_names=group_names,
                          data_a=data_cifar10,
                          data_b=_data_cifar10_1,
                          targets_a=targets_cifar10,
                          targets_b=_targets_cifar10_1,
                          split=split,
                          transform=transform)
        dataloader = DataLoader(dataset, 
                                shuffle=shuffle, 
                                batch_size=batch_size,
                                num_workers=args.num_workers)
        dataloaders.append(dataloader)
    return dataloaders


def visualize_cifar10(dataloader, num_datapoints, title, args, save,
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
        if subset:
            class_indices = subset_indices[class_indices]
        all_data_indices.extend(class_indices[:num_datapoints])
    
    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in
                     all_data_indices],
                    mean=np.mean([0.485, 0.456, 0.406]),  
                    std=np.mean([0.229, 0.224, 0.225]), nrow=8, title=title,
                    args=args, save=save, save_id=save_id, ftype=ftype)


def load_dataloaders(args, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    if args.dataset == 'cifar10e1':
        return load_cifar10e1(args, train_shuffle, transform)
    elif args.dataset == 'cifar10e2':
        return load_cifar10e2(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    return visualize_cifar10(dataloader, num_datapoints, title,
                             args, save, save_id, ftype, target_type)