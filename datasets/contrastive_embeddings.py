"""
Contrastive Embedding Dataset and DataLoader
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# For computing similarity matrices
from datasets.embeddings import get_embedding_loader

from tqdm import tqdm


class SupervisedContrastiveEmbeddingDataset(Dataset):
    def __init__(self, 
                 num_positive, 
                 num_negative,
                 num_nearest_neighbors_for_negatives,
                 source_predictions,
                 source_embeddings, 
                 source_dataloader,
                 seed,
                 balance_positive_classes=False,
                 nearest_neighbors_metric='cos_sim',
                 negative_indices_by_class=None,
                 train_sample_ratio=1,
                 replicate=0,
                 adapter=None,
                 args=None,
                 similarity_matrix=None):
        # Number of positives and negatives per anchor to sample
        if num_negative > num_nearest_neighbors_for_negatives:
            num_negative = num_nearest_neighbors_for_negatives
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_nearest_neighbors_for_negatives = num_nearest_neighbors_for_negatives
        
        # Source data
        self.source_predictions = source_predictions
        self.source_embeddings = source_embeddings
        self.source_dataloader = source_dataloader
        
        self.targets_all = source_dataloader.dataset.targets_all
        self.targets_t = self.targets_all['target']
        self.targets_s = self.targets_all['spurious']
        self.num_classes = len(np.unique(self.targets_t))
        
        self.indices_by_class_by_pred = get_indices_by_class_by_pred(
            self.targets_t,
            self.source_predictions,
            self.num_classes
        )
        
        self.indices_by_pred_by_class = get_indices_by_pred_by_class(
            self.targets_t,
            self.source_predictions,
            self.num_classes
        )
        
        # Organize data for sampling
        self.all_preds_by_all_classes = get_all_preds_by_all_classes(
            self.source_predictions, self.targets_t, self.num_classes
        )
        self.indices_by_class_by_pred = get_sample_indices_by_class_by_pred(
            self.targets_t, self.source_predictions, self.num_classes
        )
        self.balance_anchor_classes = args.balance_anchor_classes
        if self.balance_anchor_classes:
            all_num_incorrect_per_class = []
            for c, indices_by_class in enumerate(self.indices_by_class_by_pred):
                num_incorrect_per_class = []
                for p, p_indices in enumerate(indices_by_class):
                    if p != c:
                        num_incorrect_per_class.append(len(p_indices))
                num_incorrect_per_class = np.sum(num_incorrect_per_class)
                all_num_incorrect_per_class.append(num_incorrect_per_class)
            self.num_anchors = int(np.min(all_num_incorrect_per_class))
        else:
            self.num_anchors = None
            
        
        # Used for sampling negatives
        self.sample_neg_indices_by_class = negative_indices_by_class
        self.indices_by_not_class = get_all_indices_by_not_class(self.targets_t, 
                                                                 self.num_classes)
        # Nearest neighbors-based sampling
        if nearest_neighbors_metric == 'cos_sim':
            self.get_nearest_neighbors_func = get_nearest_neighbors_by_cos_sim
        elif nearest_neighbors_metric == 'dot_prod':
            self.get_nearest_neighbors_func = get_nearest_neighbors_by_dot_prod
        # Balance classes
        self.balance_positive_classes = balance_positive_classes
            
        self.seed = seed
        self.replicate = replicate
        self.train_sample_ratio = train_sample_ratio
        self.batch_size = 1 + self.num_positive + self.num_negative
        self.args = args
        
        self.similarity_matrix = similarity_matrix
        # Initialize adapter for computing nearest neighbors by adapter
        self.adapter = adapter
        self.initialize_data()
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], idx
        
    def initialize_data(self):
        self.blocks = {'data': [],
                       'target': [],
                       'target_t': [],
                       'target_s': [],
                       'sample_mask': []}
        np.random.seed(self.seed)
        
        source_embeddings_normed = (self.source_embeddings / 
                                    self.source_embeddings.norm(dim=-1, keepdim=True))
        
        for class_id in tqdm(range(self.num_classes), 
                             desc=f'Sampling contrastive batches {self.replicate}',
                             leave=False):
            try:
                # Sample anchors -> incorrect samples in each class
                anchor_indices = self.sample_anchor_indices(class_id)  # among the entire anchors
                if self.similarity_matrix is None:
                    predictions = self.source_predictions[anchor_indices]
                    similarity_matrix = get_similarity_matrix(source_embeddings_normed[anchor_indices], 
                                                              source_embeddings_normed, 
                                                              self.source_dataloader,
                                                              metric='cos_sim', 
                                                              args=self.args, 
                                                              batch_size=16384)
                else:
                    similarity_matrix = self.similarity_matrix
                    
                if self.num_nearest_neighbors_for_negatives +1 > similarity_matrix.shape[0]:
                    self.num_nearest_neighbors_for_negatives = similarity_matrix.shape[0] - 1

                if self.replicate > 50 and self.similarity_matrix is not None:
                    topk_sim, topk_indices = similarity_matrix[anchor_indices].topk(
                        k=self.num_nearest_neighbors_for_negatives +1, dim=1)
                else:
                    topk_sim, topk_indices = similarity_matrix.topk(
                        k=self.num_nearest_neighbors_for_negatives +1, dim=1)

                nearest_neighbors_original = topk_indices[:, 1:].numpy()  

                # Sample anchors, positives, negatives
                for ix, embedding in enumerate(tqdm(self.source_embeddings[anchor_indices], desc=f'Class {class_id}', leave=False)):
                    # Anchor
                    anchor_data = self.source_embeddings[anchor_indices[ix]]
                    anchor_mask = [1]
                    anchor_targets_all = {'target': self.targets_t[anchor_indices[ix]],
                                          'spurious': self.targets_s[anchor_indices[ix]]}
                    prediction = self.source_predictions[anchor_indices[ix]]

                    # Negatives
                    negative_indices = nearest_neighbors_original[ix]
                    ## Select a subset to appear in the batch
                    negative_indices = np.random.choice(negative_indices,
                                                        size=self.num_negative,
                                                        replace=False)
                    negative_data = self.source_embeddings[negative_indices]
                    ## Filter for samples with different class
                    negative_mask = self.targets_t[negative_indices] != class_id

                    negative_targets_all = {'target': self.targets_t[negative_indices],
                                            'spurious': self.targets_s[negative_indices]}

                    # Positives
                    positive_data = [] 
                    positive_mask = []
                    positive_targets_all = {'target': [],
                                            'spurious': []}  # placeholder

                    sampled_positive_outputs = self.sample_positives(class_id, prediction, 
                                                                     self.balance_positive_classes)
                    positive_data.extend(sampled_positive_outputs[0])
                    positive_data = torch.vstack(positive_data)
                    positive_mask.append(sampled_positive_outputs[1])
                    positive_mask = np.concatenate(positive_mask)

                    positive_targets_all['target'].append(sampled_positive_outputs[2]['target'])
                    positive_targets_all['spurious'].append(sampled_positive_outputs[2]['spurious'])
                    for k in positive_targets_all:
                        positive_targets_all[k] = np.concatenate(positive_targets_all[k])

                    data_block        = []
                    target_block      = []
                    target_t_block    = []
                    target_s_block    = []
                    sample_mask_block = []

                    batch_data = torch.vstack([anchor_data, 
                                               positive_data, 
                                               negative_data])
                    
                    k = 'target'
                    batch_targets_t = np.concatenate([[anchor_targets_all[k]],
                                                      positive_targets_all[k],
                                                      negative_targets_all[k]])
                    k = 'spurious'
                    batch_targets_s = np.concatenate([[anchor_targets_all[k]],
                                                      positive_targets_all[k],
                                                      negative_targets_all[k]])
                    batch_sample_masks = np.concatenate([anchor_mask,
                                                        positive_mask,
                                                        negative_mask])
                    batch_targets = torch.from_numpy(batch_targets_t)

                    self.blocks['data'].append(batch_data.reshape(self.batch_size, -1))
                    self.blocks['target'].append(batch_targets)
                    self.blocks['target_t'].append(batch_targets_t)
                    self.blocks['target_s'].append(batch_targets_s)
                    self.blocks['sample_mask'].append(batch_sample_masks)
            except ValueError:
                pass
                
        # Shuffle amongst the blocks -> keeps positives together
        shuffle_ix = np.arange(len(self.blocks['data']))
            
        np.random.seed(self.seed)
        np.random.shuffle(shuffle_ix)
        for k in self.blocks:
            assert len(self.blocks[k]) == len(shuffle_ix)
            self.blocks[k] = [self.blocks[k][ix] for ix in shuffle_ix]
            if k in ['data', 'target']:
                self.blocks[k] = torch.cat(self.blocks[k])
            else:
                self.blocks[k] = np.concatenate(self.blocks[k])
                
        self.data = self.blocks['data']
        self.targets = self.blocks['target']
        self.targets_all = {'target': self.blocks['target_t'],
                            'spurious': self.blocks['target_s'],
                            'sample_mask': self.blocks['sample_mask']}
        
    def sample_anchor_indices(self, class_id):
        all_preds_by_class = self.all_preds_by_all_classes[class_id]
        
        indices_by_pred_per_class = self.indices_by_class_by_pred[class_id]
    
        # Different predictions, but it's the same class
        incorrect_indices_by_pred = [indices_by_pred_per_class[p] for p, indices in 
                                     enumerate(indices_by_pred_per_class) 
                                     if p != class_id and len(indices) > 0]
                
        indices = np.concatenate(incorrect_indices_by_pred)
        if self.balance_anchor_classes:
            indices = np.random.choice(indices,
                                       size=self.num_anchors,
                                       replace=False)
        
        if self.train_sample_ratio < 1:
            # Reweight sampling probability to balance between classes
            p_sample = get_balanced_sampling_prob(indices, incorrect_indices_by_pred)
            indices = np.random.choice(indices,
                                       size=int(np.round(len(indices) * 
                                                         self.train_sample_ratio)),
                                       replace=True, 
                                       p=p_sample)
        return indices
    
    def sample_positives(self, class_id, pred_id, balance_classes):
        positive_data = []
        sampled_targets_all = {'target': [],
                               'spurious': []}
        positive_mask = []
        num_sampled = 0
        targets   = self.targets_t
        spurious  = self.targets_s
            
        all_preds_by_class = self.all_preds_by_all_classes[class_id]
        class_indices_by_not_pred = [all_preds_by_class[p] 
                                     for p in range(len(all_preds_by_class))
                                     if (p != pred_id and 
                                         len(all_preds_by_class[p]) > 0 and
                                         p == class_id)]
        
        indices = np.concatenate(class_indices_by_not_pred)
        
        sample_size = (len(indices) if self.num_positive > len(indices) 
                       else self.num_positive)
        if balance_classes:
            p_sample = get_balanced_sampling_prob(indices, class_indices_by_not_pred)
            _indices = np.random.choice(indices,
                                       size=sample_size,
                                       replace=True, 
                                       p=p_sample)
        else:
            _indices = np.random.choice(indices,
                                        size=sample_size,
                                        replace=False)
        positive_data.append(self.source_embeddings[_indices])
        positive_mask.append(self.targets_t[_indices] == class_id)

        sampled_targets_all['target'].append(self.targets_t[_indices])
        sampled_targets_all['spurious'].append(self.targets_s[_indices])

        num_sampled += sample_size
        # Fill rest of sampled batch with placeholders and mask out
        num_samples = self.num_positive - num_sampled
        _indices_placeholder = np.random.choice(_indices,
                                                size=num_samples,
                                                replace=True)
        positive_data.append(self.source_embeddings[_indices_placeholder])
        sampled_targets_all['target'].append(np.ones(num_samples).astype(int) * class_id)
        sampled_targets_all['spurious'].append(np.ones(num_samples).astype(int) * class_id)
        positive_mask.append(np.zeros(num_samples).astype(bool))
        
        positive_data = torch.vstack(positive_data)
        positive_mask = np.concatenate(positive_mask)
        for k in sampled_targets_all:
            sampled_targets_all[k] = np.concatenate(sampled_targets_all[k])

        return positive_data, positive_mask, sampled_targets_all
    
    
def load_supervised_contrastive_dataset(args, seed, **kwargs):
    dataset = SupervisedContrastiveEmbeddingDataset(args.num_positives,
                                                    args.num_negatives,
                                                    args.num_neighbors_for_negatives,
                                                    kwargs['source_predictions'],
                                                    kwargs['source_embeddings'],
                                                    kwargs['source_dataloader'],
                                                    seed,
                                                    args.balance_positive_classes,
                                                    args.nearest_neighbors_metric,
                                                    kwargs['negative_indices_by_class'],
                                                    args.train_sample_ratio,
                                                    args.replicate,
                                                    kwargs['adapter'],
                                                    args,
                                                    kwargs['embedding_similarity_matrix'])
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    return dataloader                
        
        
def get_balanced_sampling_prob(indices, indices_by_group):
    p_uniform = np.ones(len(indices)) / len(indices)
    # group := (y, y_hat)
    num_groups = len(indices_by_group)
    group_freqs = np.array([len(i) / len(indices) 
                            for i in indices_by_group])
    multipliers = (1. / num_groups) / group_freqs
    sample_multipliers = []
    for ix, indices_ in enumerate(indices_by_group):
        sample_multipliers.extend([multipliers[ix] for _ in indices_])
    sample_multipliers = np.array(sample_multipliers)
    p_resample = p_uniform * sample_multipliers
    return p_resample
        
        
def get_all_preds_by_all_classes(predictions, targets, num_classes):
    """
    Organizes data into samples by class and by pred
    """
    
    all_preds_by_all_classes = []
    for i in range(num_classes):
        all_preds_by_class = []
        for j in range(num_classes):
            indices = np.where(np.logical_and(
                targets == i,
                predictions == j,
            ))[0]
            all_preds_by_class.append(indices)
        all_preds_by_all_classes.append(all_preds_by_class)
    return all_preds_by_all_classes


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


def get_all_indices_by_not_class(targets, num_classes):
    all_indices_by_not_class = []
    for c in range(num_classes):
        indices = np.where(targets != c)[0]
        all_indices_by_not_class.append(indices)
    return all_indices_by_not_class


## Other ways to organize indices
def get_indices_by_pred_by_class(class_labels,
                                 predictions,
                                 num_classes):
    all_classes = range(num_classes)
    indices_by_pred_by_class = []
    
    for p in all_classes:
        indices_by_pred = []
        for c in all_classes:
            indices = np.where(np.logical_and(class_labels == c,
                                              predictions == p))[0]
            indices_by_pred.append(indices)
        indices_by_pred_by_class.append(indices_by_pred)
    return indices_by_pred_by_class


def get_indices_by_class_by_pred(class_labels,
                                 predictions,
                                 num_classes):
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


## ---------------------------------------
## Different ways to get nearest neighbors
## ---------------------------------------
def get_similarity_matrix(query_embeddings, key_embeddings, source_dataloader,
                          metric, args, batch_size=16384):

    # Assume that query_embeddings and key_embeddings are normalized
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
        return torch.vstack(all_sims)
    else:
        raise NotImplementedError 


def get_nearest_neighbors_by_cos_sim(embeddings, num_neighbors,
                                     neighbor_embeddings):
    cos_sim = []
    for ix, embedding in enumerate(embeddings):
        cos_sim.append(F.cosine_similarity(embedding.reshape(1, -1), 
                                           neighbor_embeddings))
    cos_sim = torch.vstack(cos_sim).numpy()
    sorted_indices = np.argsort(cos_sim, axis=-1)
    return np.flip(sorted_indices[:, -num_neighbors-1:-1], axis=1)


def get_nearest_neighbors_by_dot_prod(embeddings, num_neighbors,
                                      neighbor_embeddings):
    dot_prods = []
    for ix, embedding in enumerate(embeddings):
        dot_prods.append(embedding @ neighbor_embeddings.T)
    dot_prods = torch.vstack(dot_prods).numpy()
    sorted_indices = np.argsort(dot_prods, axis=-1)
    return np.flip(sorted_indices[:, -num_neighbors-1:-1], axis=1)


def get_embeddings(adapter, embedding_loader, args):
    try:
        adapter.to_device(args.device)
    except:    
        adapter.to(args.device)
    adapter.eval()
    
    if len(embedding_loader) > 200:
        pbar = tqdm(enumerate(embedding_loader), leave=False)
    else:
        pbar = enumerate(embedding_loader)
        
    all_embeddings = []
    with torch.no_grad():
        for batch_ix, data in pbar:
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            all_embeddings.append(adapter.encoder(inputs).cpu())
            inputs = inputs.cpu()
            
    all_embeddings = torch.vstack(all_embeddings)
    return all_embeddings
