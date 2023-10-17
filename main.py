"""
Training adapters on top of foundation models.
"""

import os
import sys
import copy

from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# UMAP
import umap

# CLIP
from clip import clip

from train import train_epoch as train_epoch_, evaluate as evaluate_
from train.wilds import train_epoch as train_epoch_w
from train.wilds import evaluate    as evaluate_w
from train.wilds import evaluate_wilds

from datasets import initialize_data, get_resampled_set
from datasets import get_resampled_loader_by_predictions
from datasets import get_resampled_loader_by_pred_by_class
from datasets import get_resampled_loader_by_groups

from datasets import get_indices_by_value, get_correct_indices
from datasets import update_classification_prompts
from datasets import balance_dataset_group_class, balance_dataset_class_group
from datasets import get_balanced_class_dataloader

from datasets.embeddings import EmbeddingDataset
from datasets.embeddings import get_embedding_loaders_by_split
from datasets.embeddings import get_similarity_matrix
from datasets.contrastive_embeddings import load_supervised_contrastive_dataset


from train.contrastive import ContrastiveLoss, train_contrastive_epoch
from network import get_optimizer, get_adapter

# Pretrained models
from network import load_base_model
from network.clip import evaluate_clip

from utils import initialize_save_paths, initialize_experiment
from utils.logging import summarize_acc_from_predictions
from utils.logging_wilds import summarize_acc, log_metrics, process_validation_metrics

# Final evaluation
from evaluate import get_adapter_predictions, evaluate_dataset_prediction


def get_args():
    parser = argparse.ArgumentParser(
        description='CLIP Contrastive Embedding Adapter')

    # Dataset
    parser.add_argument('--dataset', type=str, default='waterbirds')
    parser.add_argument('--dataset_size', default=200, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    # Zero-shot Model
    parser.add_argument('--load_base_model', type=str, default='')
    parser.add_argument('--zeroshot_predict_by', type=str, default='kmeans_umap',
                        choices=['text', 'kmeans', 'kmeans_umap',
                                 'ground_truth', 'proto'])

    # Classifier / Adapter Model
    parser.add_argument('--train_method', required=True,
                        choices=['linear_probe', 'adapter_mt'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--adapter_head_batch_norm', default=False,
                        action='store_true')
    parser.add_argument('--residual_connection', default=False,
                        action='store_true')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu'])
    parser.add_argument('--classification_temperature',
                        type=float, default=100)
    parser.add_argument('--num_encoder_layers', 
                        default=1, type=int)
    parser.add_argument('--zero_shot', default=False,
                        action='store_true')

    # Contrastive Training
    parser.add_argument('--contrastive_method', type=str, default='supcon',
                        choices=['simclr', 'supcon'])
    parser.add_argument('--num_positives', type=int, default=2048)
    parser.add_argument('--num_negatives', type=int, default=2048)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--balance_positive_classes', default=False,
                        action='store_true')
    parser.add_argument('--balance_anchor_classes', default=False,
                        action='store_true')
    parser.add_argument('--nearest_neighbors_metric', type=str, default='cos_sim',
                        choices=['cos_sim', 'dot_prod'])
    parser.add_argument('--num_neighbors_for_negatives', type=int, default=4096)
    parser.add_argument('--sample_pos_together',
                        default=False, action='store_true')
    parser.add_argument('--pos_num_interpolate_with', type=int, default=5)
    parser.add_argument('--pos_mc_size', type=int, default=1000)
    parser.add_argument('--pos_mc_mean_size', type=int, default=3)
    parser.add_argument('--train_sample_ratio', type=float, default=1)
    # Group size for training classification
    parser.add_argument('--group_size', type=int, default=300)

    # Hyperparams
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--bs_trn', default=128, type=int)
    parser.add_argument('--bs_val', default=128, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-5, type=float)

    # Misc.
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--replicate', default=0, type=int)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--torch_seed', default=False,
                        action='store_true')
    parser.add_argument('--verbose', default=False,
                        action='store_true')
    parser.add_argument('--num_epochs_to_end', default=0, type=int)
    # For loading Hugging Face models
    parser.add_argument('--cache_dir', 
                        default='./models/pretrained_models',
                        type=str)
    
    args = parser.parse_args()

    args.arch = args.load_base_model.split('_')[-1].replace('/', '_').replace('-', '_')
    args.group_to_class_mapping = None
    args.directory_name = 'contrastive_adapter'
        
    if 'civilcomments' in args.dataset:
        args.max_token_length = 300
    elif 'amazon' in args.dataset:
        args.max_token_length = 512
        
    if torch.cuda.is_available() and args.no_cuda is False:
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    return args


def main():
    args = get_args()
    np.random.seed(args.seed)
    if args.torch_seed:
        torch.manual_seed(args.seed)

    # Load an initial pretrained model
    # - Used to get zero-shot predictions
    # - Used to get embeddings
    args.sequence_classification_model = False
    base_model_args = args.load_base_model.split('_')
    base_model_components = load_base_model(base_model_args, args, clip=clip)
    base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions = base_model_components

    # Load data for pretrained model embeddings
    load_dataloaders, visualize_dataset = initialize_data(args)
    dataloaders_base = load_dataloaders(args, train_shuffle=False,
                                        transform=base_transform)
    
    if args.dataset in ['fmow']:
        # Refactor this to make iid_val_loader_base and iid_test_loader_base come after
        train_loader_base, iid_val_loader_base, val_loader_base, iid_test_loader_base, test_loader_base = dataloaders_base
        splits = ['train', 'iid_val', 'val', 'iid_test', 'test']
    elif args.dataset in ['iwildcam']:
        train_loader_base, val_loader_base, test_loader_base, iid_val_loader_base, iid_test_loader_base = dataloaders_base
        splits = ['train', 'val', 'test', 'id_val', 'id_test']
    else:
        train_loader_base, val_loader_base, test_loader_base = dataloaders_base
        splits = ['train', 'val', 'test']

    # Initialize other parts of experiment
    initialize_experiment(args)
    
    args.embeddings_dir = '/dfs/scratch1/mzhang/projects/flextape/embeddings/'
    args.embeddings_dir = join(args.embeddings_dir, args.dataset, args.config)
    if not os.path.exists(join(args.embeddings_dir, args.dataset)):
        os.makedirs(join(args.embeddings_dir, args.dataset))
    if not os.path.exists(join(args.embeddings_dir, args.dataset, args.config)):
        os.makedirs(join(args.embeddings_dir, args.dataset, args.config))
    
    
    if args.resample_iid:
        args.embeddings_dir = args.embeddings_dir.replace('_iid', '').split('_min')[0]
        print(args.embeddings_dir)
        if args.config not in args.embeddings_dir:
            args.embeddings_dir = join(args.embeddings_dir, args.config)
    print(args.embeddings_dir)

    # Setup training
    args.num_classes = len(args.train_classes)
    num_classes = args.num_classes
    
    # Get pretrained model 'query' (aka 'classification') embeddings
    args.text_descriptions = update_classification_prompts(args)
    query_embeddings = get_embeddings(args.text_descriptions,
                                      base_model,
                                      args,
                                      normalize=True,
                                      verbose=True)

    # Get pretrained model dataset embeddings
    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = get_dataset_embeddings(base_model,
                                                           dataloaders_base[dix],
                                                           args,
                                                           split=split)
    # Get embedding dimensions
    print(dataset_embeddings['train'].shape)
    args.base_model_dim = dataset_embeddings['train'].shape[1]
    print(f'-> Embedding dimensions: {args.base_model_dim}')
    print('---')
    # Get zero-shot predictions
    dataset_predictions = {}
    queries = query_embeddings if 'civilcomments' in args.dataset or 'amazon' in args.dataset else args.text_descriptions  # if 

    for dix, split in enumerate(splits):
        dataset_predictions[split] = get_zeroshot_predictions(dataset_embeddings[split],
                                                              queries,
                                                              args.zeroshot_predict_by,
                                                              args,
                                                              dataloaders_base[dix],
                                                              temperature=100.,
                                                              split=split,
                                                              numpy=True,
                                                              base_model=base_model)
    if args.wilds_dataset:
        eval_dicts = []
        for ix, split in enumerate(splits):
            eval_dict, eval_str = evaluate_wilds(
                torch.from_numpy(dataset_predictions[split]), 
                dataloaders_base[ix]
            )
            if 'val' in split or (args.zero_shot is True):
                print(f'-' * len(split))
                print(f'Zero-shot {split} predictions')
                print(f'-' * len(split))
                print(eval_str)
            eval_dicts.append(eval_dict)
        # Initialize evaluation functions for later training   
        train_epoch = train_epoch_w
        evaluate = evaluate_w
    else:
        for ix, split in enumerate(splits):
            print(f'-' * len(split))
            print(f'Zero-shot {split} predictions')
            print(f'-' * len(split))
            evaluate_clip(dataset_predictions[split],
                          dataloaders_base[ix],
                          verbose=True)
        # Initialize evaluation functions for later training   
        train_epoch = train_epoch_
        evaluate = evaluate_
        
    if args.zero_shot is True:
        return

    # Organize samples by prediction and correctness
    try:
        train_targets_t = dataloaders_base[0].dataset.targets_all['target']
    except Exception as e:
        print(dataloaders_base[0].dataset)
        print(dataloaders_base[0].dataset.targets_all)
        raise e

    train_indices_by_class = get_indices_by_value(train_targets_t)
    train_indices_by_pred = get_indices_by_value(dataset_predictions['train'])

    train_correct_indices, train_correct = get_correct_indices(
        dataset_predictions['train'], train_targets_t
    )

    # Prepare embedding dataloaders
    embeddings = [dataset_embeddings[split] for split in dataset_embeddings]
    indices = [[np.arange(len(e))] for e in embeddings]
    dataloaders = get_embedding_loaders_by_split(embeddings,
                                                 indices,
                                                 dataloaders_base,
                                                 train_shuffle=False,
                                                 args=args)
    dataloaders = [dataloader[0] for dataloader in dataloaders]
    if args.dataset == 'fmow':
        train_loader, iid_val_loader, val_loader, iid_test_loader, test_loader = dataloaders
    elif args.dataset == 'iwildcam':
        train_loader, val_loader, test_loader, iid_val_loader, iid_test_loader = dataloaders
    else:
        train_loader, val_loader, test_loader = dataloaders
        
    # Organize by class
    train_indices_by_class = []
    for t in np.unique(dataloaders_base[splits.index('train')].dataset.targets):
        train_indices_by_class.append(np.where(dataloaders_base[splits.index('train')].dataset.targets == t)[0])
        
    eval_train_loader = train_loader
    classifier_criterion = nn.CrossEntropyLoss()
    num_positives = args.num_positives
    contrastive_criterion = ContrastiveLoss(args.temperature,
                                            args.num_negatives,
                                            num_positives)
    contrastive_args = {'source_predictions': dataset_predictions['train'],
                        'source_embeddings': dataset_embeddings['train'],
                        'source_dataloader': train_loader,
                        'negative_indices_by_class': None,
                        'adapter': None}
    
    contrastive_args['source_embeddings_normed'] = (
        contrastive_args['source_embeddings'] / 
        contrastive_args['source_embeddings'].norm(dim=-1, keepdim=True))
    
    try:   # Precompute similarity matrix for sampling negatives
        similarity_matrix = get_similarity_matrix(
            contrastive_args['source_embeddings_normed'],
            contrastive_args['source_embeddings_normed'],
            contrastive_args['source_dataloader'],
            metric='cos_sim',
            args=args,
            batch_size=16384 * 2,
            split='train'
        )
    except:
        similarity_matrix = None
    contrastive_args['embedding_similarity_matrix'] = similarity_matrix

    # Training
    args.best_loss = 1e10
    args.best_loss_epoch = -1
    args.best_avg_acc = -1
    args.best_avg_acc_epoch = -1
    args.best_robust_acc = -1
    args.best_robust_acc_epoch = -1
    
    args.best_iid_loss = 1e10
    args.best_iid_loss_epoch = -1
    args.best_iid_avg_acc = -1
    args.best_iid_avg_acc_epoch = -1
    args.best_iid_robust_acc = -1
    args.best_iid_robust_acc_epoch = -1

    # Remove pretrained model
    try:
        base_model.cpu()
    except:
        pass
    del base_model
        
    
    model = get_adapter(num_classes, args, query_embeddings)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    args.num_adapter_parameters = params
    optimizer = get_optimizer(model, args)
    
    print(f'-> Base zero-shot model = {args.load_base_model}')
    print(f'   - number of parameters:   {args.num_base_model_parameters}')
    print(f'-> Adapter train method = {args.train_method}')
    print(f'   - number of parameters:   {args.num_adapter_parameters}')
    print(f'   - input_dim:              {args.base_model_dim}')
    print(f'   - hidden_dim:             {args.hidden_dim}')
    print(f'   - batch_norm:             {args.adapter_head_batch_norm}')
    print(f'   - residual connection:    {args.residual_connection}')
    print(f'   - activation function:    {args.activation}')
    contrastive_batch_size = 2 + args.num_negatives if args.contrastive_method == 'simclr' else 1 + \
        args.num_positives + args.num_negatives
    try:
        print(f'   - prompt dot product:     {model.queries[0] @ model.queries[1].T:.4f}')
    except:
        pass
    print(f'   - contrastive_batch_size: {contrastive_batch_size}')
    print(f'-> Classification prompts ({args.num_classes}):')
    for prompt in args.text_descriptions:
        print(f'   - {prompt}')
    print(f'-> Experiment: {args.experiment_name}')
    
    # Keep a regular dataloader (not contrastive)
    train_loader_ = DataLoader(train_loader.dataset, 
                               shuffle=True,
                               batch_size=args.bs_trn, 
                               num_workers=args.num_workers)
    
    try:
        dataset_predictions['train'] = dataset_predictions['train'].numpy()
    except:
        pass
    train_loader_incorrect = get_resampled_loader_by_predictions(
        dataset_predictions['train'], train_loader.dataset,
        resample='incorrect', seed=args.seed, args=args,
        batch_size=128
    )

    pbar = tqdm(range(args.max_epochs))
    
    best_metric_counter = 0
    stopped_early = False
    
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description(f'Epoch: {epoch}')
        else:
            (train_loss, train_avg_acc, train_min_acc) = train_m
            (val_loss, val_avg_acc, val_min_acc) = val_m
            pbar.set_description(f'Epoch: {epoch} | Train Loss: {train_loss:.3f}, Avg Acc: {train_avg_acc:.2f}%, Robust Acc: {train_min_acc:.2f}% | Val Loss: {val_loss:.3f}, Avg Acc: {val_avg_acc:.2f}%, Robust Acc: {val_min_acc:.2f}%, Best Avg Acc: {args.best_avg_acc:.2f}%, Best Robust Acc: {args.best_robust_acc:.2f}%, Best Robust Epoch: {args.best_robust_acc_epoch}')

        # Just train with the contrastive component (no cross-entropy supervision)
        if args.replicate in [0]:
            try:
                del contrastive_train_loader
            except:
                pass
            contrastive_train_loader = load_supervised_contrastive_dataset(
                args, epoch + args.seed, **contrastive_args
            )
            train_metrics = train_contrastive_epoch(model,
                                                    optimizer,
                                                    classifier_criterion,
                                                    contrastive_criterion,
                                                    contrastive_train_loader,
                                                    args)
        # Default training procedure
        # - Every even epoch, train with contrastive
        # - Every odd epoch, train with group-balanced cross-entropy (inferred groups)
        elif args.replicate in [1] and ((epoch + 1) % 2 == 0):
            try:
                dataset_predictions['train'] = dataset_predictions['train'].numpy()
            except:
                pass
            try:
                del train_loader_rs
            except:
                pass
            train_loader_rs = get_resampled_loader_by_pred_by_class(
                dataset_predictions['train'], train_loader.dataset,
                resample='group_size', seed=epoch + args.seed, args=args,
                batch_size=128,
            )
            train_metrics = train_epoch(model, train_loader_rs,
                                        optimizer, classifier_criterion,
                                        args, evaluate=False)
                
        elif args.replicate in [2] and ((epoch + 1) % 2 == 0):
            try:
                dataset_predictions['train'] = dataset_predictions['train'].numpy()
            except:
                pass
            try:
                del train_loader_rs
            except:
                pass
            dataset_rs = balance_dataset_class_group(dataset_predictions['train'],
                                                     train_loader.dataset,
                                                     epoch+args.seed)
            
            if len(dataset_rs) % args.bs_trn == 1:
                drop_last=True
            else:
                drop_last = False

            train_loader_rs = DataLoader(dataset_rs, batch_size=args.bs_trn, shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=drop_last)
            train_metrics = train_epoch(model, train_loader_rs,
                                        optimizer, classifier_criterion,
                                        args, evaluate=False)

        # Only train with samples that zero-shot misclassified
        elif args.replicate in [3]:   
            train_metrics = train_epoch(model, train_loader_incorrect,
                                        optimizer, classifier_criterion,
                                        args)
        # Train with ERM    
        elif args.replicate in [4]: 
            train_metrics = train_epoch(model, train_loader_,
                                        optimizer, classifier_criterion,
                                        args)
        # Train with upsampling zero-shot misclassified points
        elif args.replicate in [5]:
            try:
                del train_loader_rs
            except:
                pass
            train_loader_rs = get_resampled_loader_by_predictions(
                dataset_predictions['train'], train_loader.dataset,
                resample='upsample', seed=epoch + args.seed, args=args
            )
            if epoch == 0:
                print(np.unique(train_loader_rs.dataset.targets_all['group_idx'],
                                return_counts=True))
            
            train_metrics = train_epoch(model, train_loader_rs,
                                        optimizer, classifier_criterion,
                                        args)
        # Train with subsampling zero-shot misclassified points
        elif args.replicate in [6]:
            try:
                del train_loader_rs
            except:
                pass

            train_loader_rs = get_resampled_loader_by_predictions(
                dataset_predictions['train'], train_loader.dataset,
                resample='subsample', seed=epoch + args.seed, args=args
            )
            if epoch == 0:
                print(np.unique(train_loader_rs.dataset.targets_all['group_idx'],
                                return_counts=True))
            
            train_metrics = train_epoch(model, train_loader_rs,
                                        optimizer, classifier_criterion,
                                        args)
        else:  # Replicates 1, 2
            # Prepare contrastive dataloader
            contrastive_train_loader = load_supervised_contrastive_dataset(
                args, epoch + args.seed, **contrastive_args
            )
            train_metrics = train_contrastive_epoch(model,
                                                    optimizer,
                                                    classifier_criterion,
                                                    contrastive_criterion,
                                                    contrastive_train_loader,
                                                    args)
        train_metrics = evaluate(model, train_loader, classifier_criterion, args)
        val_metrics = evaluate(model, val_loader, classifier_criterion, args)
        test_metrics = evaluate(model, test_loader, classifier_criterion, args)
        
        train_m, val_m = log_metrics(train_metrics, val_metrics,
                                     test_metrics, epoch,
                                     dataset_ix=0, args=args)
        train_method = args.train_method_verbose.replace('/', '_').replace('-', '_')
        loss_m, avg_acc_m, min_acc_m = process_validation_metrics(model, val_metrics, epoch,
                                                                  train_method, args,
                                                                  best_loss=args.best_loss,
                                                                  best_avg_acc=args.best_avg_acc,
                                                                  best_robust_acc=args.best_robust_acc,
                                                                  best_loss_epoch=args.best_loss_epoch,
                                                                  best_avg_acc_epoch=args.best_avg_acc_epoch,
                                                                  best_robust_acc_epoch=args.best_robust_acc_epoch)
        
        # End experiment early if best validation metric hasn't changed
        if min_acc_m[-1] != args.best_robust_acc_epoch:
            best_metric_counter = 0
        else:
            best_metric_counter += 1
        
        args.best_loss, args.best_loss_epoch = loss_m
        args.best_avg_acc, args.best_avg_acc_epoch = avg_acc_m
        args.best_robust_acc, args.best_robust_acc_epoch = min_acc_m
        
        if args.dataset in ['fmow']:
            iid_val_metrics  = evaluate(model, iid_val_loader, classifier_criterion, args)
            iid_test_metrics = evaluate(model, iid_test_loader, classifier_criterion, args)
            train_m, iid_val_m = log_metrics(train_metrics, iid_val_metrics,
                                             iid_test_metrics, epoch,
                                             dataset_ix=0, args=args,
                                             save_train_metrics=False,
                                             train_split='train_iid', 
                                             val_split='val_iid', 
                                             test_split='test_iid')
            loss_m, avg_acc_m, min_acc_m = process_validation_metrics(model, val_metrics, epoch,
                                                                      train_method, args,
                                                                      save_id='iid',
                                                                      best_loss=args.best_iid_loss,
                                                                      best_avg_acc=args.best_iid_avg_acc,
                                                                      best_robust_acc=args.best_iid_robust_acc,
                                                                      best_loss_epoch=args.best_iid_loss_epoch,
                                                                      best_avg_acc_epoch=args.best_iid_avg_acc_epoch,
                                                                      best_robust_acc_epoch=args.best_iid_robust_acc_epoch)
            args.best_iid_loss, args.best_iid_loss_epoch = loss_m
            args.best_iid_avg_acc, args.best_iid_avg_acc_epoch = avg_acc_m
            args.best_iid_robust_acc, args.best_iid_robust_acc_epoch = min_acc_m

        if (epoch + 1) % 10 == 0:
            try:
                df = pd.DataFrame(args.results_dict)
                df.to_csv(args.results_path)
            except Exception as e:
                for key in args.results_dict:
                    print(f'len(args.results_dict[{key}]): {len(args.results_dict[key])}')
                raise e
            
        if best_metric_counter > args.num_epochs_to_end and args.num_epochs_to_end != 0:
            stopped_early = True
            break
    
    if stopped_early:
        print(f'-> Stopping experiment after {epoch + 1} epochs...')
        
    try:
        df = pd.DataFrame(args.results_dict)
        df.to_csv(args.results_path)
    except Exception as e:
        pass

    # Get max validation output
    metric = 'val_robust_acc' if args.replicate != 7 else 'val_avg_acc'
    max_robust_acc_ix = np.argmax(df['val_robust_acc'])
    max_epoch = df['epoch'].iloc[max_robust_acc_ix]
    print(f'-> Best validation metrics (Epoch {max_epoch}):')
    print(df[['val_avg_acc', 'val_robust_acc',
              'test_avg_acc', 'test_robust_acc']].iloc[max_robust_acc_ix])
    if args.dataset in ['fmow', 'iwildcam']:
        max_robust_acc_ix = np.argmax(df[f'val_{args.save_id}_robust_acc'])
        max_epoch = df['epoch'].iloc[max_robust_acc_ix]
        print(f'Best {args.save_id} validation metrics (Epoch {max_epoch}):')
        print(df[[f'val_{args.save_id}_avg_acc', f'val_{args.save_id}_robust_acc',
                  f'test_{args.save_id}_avg_acc', f'test_{args.save_id}_robust_acc']].iloc[max_robust_acc_ix])
    # Final evaluation
    model = get_adapter(num_classes, args, query_embeddings)
    print(f'=== Best model by robust validation performance ===')
    model.load_state_dict(torch.load(args.best_robust_acc_model_path))
    for dix, split in enumerate(splits):
        print('-' * 5, split, '-' * 5)
        embedding_loader = dataloaders[dix]
        predictions = get_adapter_predictions(model, 
                                              embedding_loader,
                                              args.device)
        avg_acc, wg_acc = evaluate_dataset_prediction(predictions,
                                                      embedding_loader,
                                                      args, verbose=True)
    model.load_state_dict(torch.load(args.best_avg_acc_model_path))
    print(f'=== Best model by average validation performance ===')
    for dix, split in enumerate(splits):
        print('-' * 5, split, '-' * 5)
        embedding_loader = dataloaders[dix]
        predictions = get_adapter_predictions(model, 
                                              embedding_loader,
                                              args.device)
        avg_acc, wg_acc = evaluate_dataset_prediction(predictions,
                                                      embedding_loader,
                                                      args, verbose=True)


if __name__ == '__main__':
    main()
