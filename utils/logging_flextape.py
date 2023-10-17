"""
Initialize and log results for every adapter
"""

import os
import numpy as np
import torch

from os.path import join
from copy import deepcopy

from utils.logging_wilds import summarize_acc


def initialize_results_dicts(num_adapters, results_dict):
    all_results_dicts = []
    all_best_results_dicts = []
    
    for adapter in range(num_adapters):
        all_results_dicts.append(deepcopy(results_dict))
        best_results_dict = {
            'best_loss': 1e10,
            'best_loss_epoch': -1,
            'best_avg_acc': -1,
            'best_avg_acc_epoch': -1,
            'best_robust_acc': -1,
            'best_robust_acc_epoch': -1,
            'best_iid_loss': 1e10,
            'best_iid_loss_epoch': -1,
            'best_iid_avg_acc': -1,
            'best_iid_avg_acc_epoch': -1,
            'best_iid_robust_acc': -1,
            'best_iid_robust_acc_epoch': -1,
        }
        all_best_results_dicts.append(best_results_dict)
    return all_results_dicts, all_best_results_dicts


def log_metrics(train_metrics, val_metrics, test_metrics, 
                all_results_dicts, prediction_id, epoch,
                dataset_ix=0, args=None, save_train_metrics=True,
                train_split='train', val_split='val', test_split='test'):
    assert args is not None
    if args.wilds_dataset:
        train_loss, correct, total, train_eval_dict, eval_str = train_metrics
        train_avg_acc = train_eval_dict[args.average_group_key] * 100
        train_min_acc = train_eval_dict[args.worst_group_key] * 100
        if args.verbose:
            print(train_eval_str)
            
        val_loss, correct, total, val_eval_dict, eval_str = val_metrics
        val_avg_acc = val_eval_dict[args.average_group_key] * 100
        val_min_acc = val_eval_dict[args.worst_group_key] * 100
        if args.verbose:
            print(eval_str)
            
        test_loss, correct, total, test_eval_dict, eval_str = test_metrics
        test_avg_acc = test_eval_dict[args.average_group_key] * 100
        test_min_acc = test_eval_dict[args.worst_group_key] * 100
        if args.verbose:
            print(eval_str)
            
    else:
        train_loss, correct, total, correct_by_groups, total_by_groups = train_metrics
        train_avg_acc, train_min_acc = summarize_acc(correct_by_groups,
                                                     total_by_groups,
                                                     stdout=args.verbose)
        val_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
        val_avg_acc, val_min_acc = summarize_acc(correct_by_groups,
                                                 total_by_groups,
                                                 stdout=args.verbose)
        test_loss, correct, total, correct_by_groups, total_by_groups = test_metrics
        test_avg_acc, test_min_acc = summarize_acc(correct_by_groups,
                                                   total_by_groups,
                                                   stdout=args.verbose)
    if save_train_metrics is True:
        all_results_dicts[prediction_id]['epoch'].append(epoch)
        all_results_dicts[prediction_id]['dataset_ix'].append(dataset_ix)
        all_results_dicts[prediction_id][f'{train_split}_loss'].append(train_loss)
        all_results_dicts[prediction_id][f'{train_split}_avg_acc'].append(train_avg_acc)
        all_results_dicts[prediction_id][f'{train_split}_robust_acc'].append(train_min_acc)

    all_results_dicts[prediction_id][f'{val_split}_loss'].append(val_loss)
    all_results_dicts[prediction_id][f'{val_split}_avg_acc'].append(val_avg_acc)
    all_results_dicts[prediction_id][f'{val_split}_robust_acc'].append(val_min_acc)

    all_results_dicts[prediction_id][f'{test_split}_loss'].append(test_loss)
    all_results_dicts[prediction_id][f'{test_split}_avg_acc'].append(test_avg_acc)
    all_results_dicts[prediction_id][f'{test_split}_robust_acc'].append(test_min_acc)
    
    train_metrics = (train_loss, train_avg_acc, train_min_acc)
    val_metrics = (val_loss, val_avg_acc, val_min_acc)
    return train_metrics, val_metrics


def process_validation_metrics(model, val_metrics, epoch, train_method, args,
                               all_results_dicts, all_best_results_dicts,
                               prediction_id, model_id=None, save_id=None):
    
    
    if args.wilds_dataset:
        val_loss, correct, total, val_eval_dict, eval_str = val_metrics
        avg_acc = val_eval_dict[args.average_group_key] * 100
        min_acc = val_eval_dict[args.worst_group_key] * 100
    else:
        val_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
        avg_acc, min_acc = summarize_acc(correct_by_groups,
                                         total_by_groups,
                                         stdout=False)
        
    if model_id is None:
        model_id = f'p_{prediction_id:03d}'
    else:
        model_id = f'p_{prediction_id:03d}_{model_id}'
        
    if save_id is None:
        save_id = ''
    else:
        save_id = f'_{save_id}'
        
    best_loss = all_best_results_dicts[prediction_id][f'best_loss{save_id}']    
    best_loss_epoch = all_best_results_dicts[prediction_id][f'best_loss{save_id}_epoch']    
    best_avg_acc = all_best_results_dicts[prediction_id][f'best_avg_acc{save_id}']
    best_avg_acc_epoch = all_best_results_dicts[prediction_id][f'best_avg_acc{save_id}_epoch']
    best_robust_acc = all_best_results_dicts[prediction_id][f'best_robust_acc{save_id}']
    best_robust_acc_epoch = all_best_results_dicts[prediction_id][f'best_robust_acc{save_id}_epoch']
        
    if val_loss < all_best_results_dicts[prediction_id]['best_loss'] or epoch == 0:
        best_loss_epoch = epoch
        best_loss = val_loss
        all_best_results_dicts[prediction_id][f'best_loss{save_id}_model_path'] = join(
            args.model_dir, 
            f'm-best_loss-{model_id}{save_id}-{train_method}-r={args.replicate}-s={args.seed}.pt'
        )
        all_best_results_dicts[prediction_id][f'best_loss{save_id}_model_path'] = all_best_results_dicts[prediction_id]['best_loss_model_path'].replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')
        torch.save(model.state_dict(), all_best_results_dicts[prediction_id]['best_loss_model_path'])
    
    all_results_dicts[prediction_id][f'best_loss{save_id}_epoch'].append(best_loss_epoch)

    if avg_acc > all_best_results_dicts[prediction_id]['best_avg_acc'] or epoch == 0:
        best_avg_acc_epoch = epoch
        best_avg_acc = avg_acc
        all_best_results_dicts[prediction_id][f'best_avg_acc{save_id}_model_path'] = join(
            args.model_dir,
            f'm-best_acc-{model_id}{save_id}-{train_method}-r={args.replicate}-s={args.seed}.pt'
        )
        all_best_results_dicts[prediction_id][f'best_avg_acc{save_id}_model_path'] = all_best_results_dicts[prediction_id][f'best_avg_acc{save_id}_model_path'].replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')
        torch.save(model.state_dict(), all_best_results_dicts[prediction_id]['best_avg_acc_model_path'])
    all_results_dicts[prediction_id][f'best_acc{save_id}_epoch'].append(best_avg_acc_epoch)
    
    if min_acc > all_best_results_dicts[prediction_id]['best_robust_acc'] or epoch == 0:
        best_robust_acc_epoch = epoch
        best_robust_acc = min_acc
        all_best_results_dicts[prediction_id][f'best_robust_acc{save_id}_model_path'] = join(
            args.model_dir,
            f'm-best_acc_r-{model_id}{save_id}-{train_method}-r={args.replicate}-s={args.seed}.pt'
        )
        all_best_results_dicts[prediction_id][f'best_robust_acc{save_id}_model_path'] = all_best_results_dicts[prediction_id][f'best_robust_acc{save_id}_model_path'].replace('facebook_', '').replace('supcon', 'sc')
        torch.save(model.state_dict(), all_best_results_dicts[prediction_id]['best_robust_acc_model_path'])

    all_results_dicts[prediction_id][f'best_robust_acc{save_id}_epoch'].append(best_robust_acc_epoch)

    return (best_loss, best_loss_epoch), (best_avg_acc, best_avg_acc_epoch), (best_robust_acc, best_robust_acc_epoch)
#     except Exception as e:
#         raise e
#         return (best_loss, best_loss_epoch), (best_avg_acc, best_avg_acc_epoch), (None, None)
    
    



