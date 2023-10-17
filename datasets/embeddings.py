"""
Dataset and Dataloader for loading embeddings
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, indices, source_dataset):
        self.embeddings = embeddings
        self.indices = indices
        
        source_targets = source_dataset.targets_all['target']
        source_spurious = source_dataset.targets_all['spurious']
        
        try:
            source_group_idx = source_dataset.targets_all['group_idx']
            group_idx = source_group_idx[indices]
        except KeyError:
            group_idx = np.ones(len(indices))
        
        self.data = self.embeddings[indices]
        self.targets = source_dataset.targets[indices]
        
        self.targets_all = {'target': source_targets[indices],
                            'spurious': source_spurious[indices],
                            'group_idx': group_idx}
        
        # For WILDS datasets
        self.source_dataset = source_dataset
        try:
            self.eval = source_dataset.eval
            self.y_array = source_dataset.y_array
            self.metadata_array = source_dataset.metadata_array
        except:
            pass
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], idx
    

def get_embedding_loader(embeddings, source_dataset, shuffle, args,
                         batch_size=None):
    dataset = EmbeddingDataset(embeddings, np.arange(len(embeddings)), 
                               source_dataset)
    batch_size = args.bs_trn if batch_size is None else batch_size
    dataloader = DataLoader(dataset, 
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=0)  # args.num_workers)
    return dataloader
    
    
    
def get_embedding_loaders_by_split(embeddings, indices, source_dataloaders,
                                   train_shuffle, args):
    """
    Args:
    - embeddings: [train_embeddings, val_embeddings, test_embeddings]
    - indices: [(train_indices_0, ..., train_indices_n), 
                (val_indices_0, ..., val_indices_n),
                (test_indices_0, ..., test_indices_n)]
    - source_dataloaders: [train_loader, val_loader, test_loader]  
    Returns:
    - dataloaders: [train_loaders[...], val_loaders[...], test_loaders[...]]
    """
    dataloaders = []
    
    for ix in range(len(embeddings)):
        dataloaders_by_split = []
        shuffle = train_shuffle if ix == 0 else False
        batch_size = args.bs_trn if ix == 0 else args.bs_val
        for indices_ in indices[ix]:
            dataset = EmbeddingDataset(embeddings[ix], indices_, 
                                       source_dataloaders[ix].dataset)
            dataloader = DataLoader(dataset, 
                                    shuffle=train_shuffle,
                                    batch_size=args.bs_trn,
                                    num_workers=args.num_workers)
            dataloaders_by_split.append(dataloader)
        dataloaders.append(dataloaders_by_split)
    
    return dataloaders


def get_similarity_matrix(query_embeddings, key_embeddings, source_dataloader,
                          metric, args, batch_size=16384, split='train', 
                          verbose=True, retrieve=True):
    # Assume that query_embeddings and key_embeddings are normalized
    embeddings_dir = args.embeddings_dir
    load_base_model = args.load_base_model.replace('/', '_')
    embedding_fname = f'sim_matrix-d={args.dataset}-s={split}-c={args.config}-m={load_base_model}.pt'
    embedding_path = os.path.join(embeddings_dir, embedding_fname)
    if os.path.exists(embedding_path) and retrieve:
        if verbose:
            print(f'-> Retrieving similarity matrix from {embedding_path}!')
        embeddings = torch.load(embedding_path)
        return torch.load(embedding_path)
    else:
        print(f'-> Similarity matrix from {embedding_path} not found. Computing...')
    
    if metric == 'cos_sim':
        query_loader = get_embedding_loader(query_embeddings,
                                            source_dataloader.dataset,
                                            shuffle=False,
                                            args=args,
                                            batch_size=batch_size)
        key_loader = get_embedding_loader(key_embeddings,
                                          source_dataloader.dataset,
                                          shuffle=False,
                                          args=args,
                                          batch_size=batch_size)
        all_sims = []
        for ix, data_i in enumerate(tqdm(query_loader, leave=False, 
                                         desc=f'Computing similarity by row')):
            data_i = data_i[0]
            data_i_sims = []
            for jx, data_j in enumerate(tqdm(key_loader, leave=False, 
                                             desc=f'Computing similarity by col')):
                data_i = data_i.to(args.device)
                data_j = data_j[0].to(args.device)
                sim = torch.matmul(data_i, data_j.T).cpu()
                data_i_sims.append(sim)
                data_i = data_i.cpu()
                data_j = data_j.cpu()
            all_sims.append(torch.hstack(data_i_sims))
        try:
            similarity_matrix = torch.vstack(all_sims)
            torch.save(similarity_matrix, embedding_path)
            print(f'-> Similarity matrix saved to {embedding_path}!')
            print(f'   - similarity_matrix.shape: {similarity_matrix.shape}')
        except RuntimeError:
            for ix, sim in enumerate(all_sims):
                e_path = embedding_path.replace('.pt', f'chunk={ix}.pt')
                torch.save(sim, e_path)
                print(f'-> Similarity matrix chunk {ix} saved to {e_path}!')
            return all_sims
        return similarity_matrix
    else:
        raise NotImplementedError 