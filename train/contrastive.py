"""
Training functions and loss objective for contrastive adapters
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from train import prep_metrics




class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, num_negatives, num_positives=1):
        super().__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=1)
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        
    def forward(self, features, masks):
        # Assume features = [anc, positives, negatives]
        # masks = masks[self.num_positives + 1:]
        negative_indices = np.arange(self.num_positives + 1, len(features))
        negative_indices = negative_indices[
            np.where(masks[self.num_positives + 1:] == 1)]
        negative_indices = np.concatenate([[0], negative_indices])
        exp_neg = self.compute_exp_sim(features[negative_indices])
        
        positive_indices = np.arange(0, self.num_positives + 1)
        positive_indices = positive_indices[
            np.where(masks[:self.num_positives + 1] == 1)]
        
        exp_pos = self.compute_exp_sim(features[positive_indices])
        log_probs = (torch.log(exp_pos) - 
                     torch.log(exp_pos + exp_neg.sum(0, keepdim=True)))
        loss = -1 * log_probs
        return loss
        
    def compute_exp_sim(self, features):
        sim = self.sim(features[0].view(1, -1), features[1:])
        exp_sim = torch.exp(torch.div(sim, self.temperature))
        return exp_sim

    
def train_contrastive_epoch(model, optimizer, classifier_criterion, 
                            contrastive_criterion, dataloader, args,
                            query_embeddings=None):
    running_loss, correct, total, targets_s, targets_t, correct_by_groups, total_by_groups, losses_by_groups = prep_metrics(dataloader, args)
    targets_t = dataloader.dataset.targets_all['target']
    targets_s = dataloader.dataset.targets_all['spurious']
    num_t = len(np.unique(targets_t))
    num_s = len(np.unique(targets_s))
    
    sample_masks = dataloader.dataset.targets_all['sample_mask']
    running_loss_con = 0
    try:
        model.to_device(args.device)
    except:    
        model.to(args.device)
    model.train()
    model.zero_grad()
    
    pbar = tqdm(dataloader, leave=False)
    for batch_ix, data in enumerate(pbar):
        inputs, labels, data_ix = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        labels_target = np.array([targets_t[dix] for dix in data_ix])
        labels_spurious = np.array([targets_s[dix] for dix in data_ix])
        contrastive_mask = np.array([sample_masks[dix] for dix in data_ix])
        
        with torch.set_grad_enabled(True):
            with torch.autograd.set_detect_anomaly(True):
                y_hat, z = model.forward(inputs, return_hidden=True)

                loss = contrastive_criterion(z, contrastive_mask)
                loss = loss.mean()
                loss.backward()

                if ((batch_ix + 1) % args.bs_trn == 0 or 
                    batch_ix + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Save performance
        _, predicted = torch.max(y_hat.data, 1)
        running_loss += loss.item()
        
        inputs = inputs.cpu()
        loss = loss.cpu()
        
        
        pbar.set_description(
            f'Batch Idx: ({batch_ix}/{len(dataloader)}) | Contrastive Loss: {running_loss / (batch_ix + 1):.3f}')
    try:
        model.to_device(torch.device('cpu'))
    except:    
        model.cpu()
    return running_loss, correct, total, correct_by_groups, total_by_groups
