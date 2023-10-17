import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_resampled_loader_by_predictions(predictions, initial_dataset, 
                                        resample, seed, args):
    dataset = resample_dataset_by_predictions(predictions,
                                              initial_dataset,
                                              resample,
                                              args.group_size, 
                                              seed)
    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=args.bs_trn,
                            num_workers=args.num_workers)
    return dataloader


def resample_dataset_by_predictions(predictions, initial_dataset, 
                                    resample, group_size=None,
                                    seed=42):
    np.random.seed(seed)
    # predictions = initial_dataset.targets_all['spurious']
    targets = initial_dataset.targets_all['target']
    all_sample_indices = []
    for t in np.unique(targets):
        t_indices = np.where(targets == t)[0]
#         y_indices 
    
    
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
            # Hack for now - assume the predictions get all groups
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
    # print(f'Resampled dataset size = {ratio:.1f}% of original') 
    group_ix, group_counts = np.unique(dataset_p.targets_all['group_idx'], return_counts=True)
    # print(' | '.join([f'Group {group_ix[ix]} count: {group_counts[ix]}' for ix in range(len(group_ix))]))
    return dataset_p