import os
import numpy as np
import torch

from os.path import join


### Logging
def summarize_acc(correct_by_groups, total_by_groups, 
                  stdout=True, return_groups=False):
    all_correct = 0
    all_total = 0
    min_acc = 101.
    min_correct_total = [None, None]
    groups_accs = np.zeros([len(correct_by_groups), 
                            len(correct_by_groups[-1])])
    if stdout:
        print('Accuracies by groups:')
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            acc = a_group / total_by_groups[yix][aix] * 100
            groups_accs[yix][aix] = acc
            # Don't report min accuracy if there's no group datapoints
            if acc < min_acc and total_by_groups[yix][aix] > 0:
                min_acc = acc
                min_correct_total[0] = a_group
                min_correct_total[1] = total_by_groups[yix][aix]
            if stdout:
                print(
                    f'{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}')
            all_correct += a_group
            all_total += total_by_groups[yix][aix]
    if stdout:
        average_str = f'Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
        robust_str = f'Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
        print('-' * len(average_str))
        print(average_str)
        print(robust_str)
        print('-' * len(average_str))
        
    avg_acc = all_correct / all_total * 100
        
    if return_groups:
        return avg_acc, min_acc, groups_accs
    return avg_acc, min_acc 


def summarize_acc_from_predictions(predictions, dataloader,
                                   args, stdout=True):
    targets_t = dataloader.dataset.targets_all['target']
    targets_s = dataloader.dataset.targets_all['spurious']
    
    correct_by_groups = np.zeros([args.num_classes,
                                  args.num_classes])
    total_by_groups = np.zeros(correct_by_groups.shape)
    
    all_correct = (predictions == targets_t)
    for ix, s in enumerate(targets_s):
        y = targets_t[ix]
        correct_by_groups[int(y)][int(s)] += all_correct[ix]
        total_by_groups[int(y)][int(s)] += 1
    return summarize_acc(correct_by_groups, total_by_groups,
                         stdout=stdout)


def log_metrics(train_metrics, val_metrics, test_metrics, epoch,
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
        args.results_dict['epoch'].append(epoch)
        args.results_dict['dataset_ix'].append(dataset_ix)
        args.results_dict[f'{train_split}_loss'].append(train_loss)
        args.results_dict[f'{train_split}_avg_acc'].append(train_avg_acc)
        args.results_dict[f'{train_split}_robust_acc'].append(train_min_acc)

    args.results_dict[f'{val_split}_loss'].append(val_loss)
    args.results_dict[f'{val_split}_avg_acc'].append(val_avg_acc)
    args.results_dict[f'{val_split}_robust_acc'].append(val_min_acc)

    args.results_dict[f'{test_split}_loss'].append(test_loss)
    args.results_dict[f'{test_split}_avg_acc'].append(test_avg_acc)
    args.results_dict[f'{test_split}_robust_acc'].append(test_min_acc)
    
    train_metrics = (train_loss, train_avg_acc, train_min_acc)
    val_metrics = (val_loss, val_avg_acc, val_min_acc)
    return train_metrics, val_metrics


def process_validation_metrics(model, val_metrics, epoch, train_method, args,
                               save_id=None, best_loss=None, best_avg_acc=None,
                               best_robust_acc=None, best_loss_epoch=None, 
                               best_avg_acc_epoch=None,
                               best_robust_acc_epoch=None):
    if args.wilds_dataset:
        val_loss, correct, total, val_eval_dict, eval_str = val_metrics
        avg_acc = val_eval_dict[args.average_group_key] * 100
        min_acc = val_eval_dict[args.worst_group_key] * 100
    else:
        val_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
        avg_acc, min_acc = summarize_acc(correct_by_groups,
                                         total_by_groups,
                                         stdout=False)
        
    if save_id is None:
        save_id = ''
    else:
        save_id = f'_{save_id}'
    if val_loss < best_loss or epoch == 0:
        best_loss_epoch = epoch
        best_loss = val_loss
        
        if f'-o={args.optimizer}-lr={args.lr}' in train_method:
            args.best_loss_model_path = join(args.model_dir, 
                                             f'm-best_loss{save_id}-{train_method}-r={args.replicate}-s={args.seed}.pt')
        else:
            args.best_loss_model_path = join(args.model_dir, 
                                             f'm-best_loss{save_id}-{train_method}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}.pt')

        args.best_loss_model_path = args.best_loss_model_path.replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')
        torch.save(model.state_dict(), args.best_loss_model_path)
    args.results_dict[f'best_loss{save_id}_epoch'].append(best_loss_epoch)

    if avg_acc > best_avg_acc or epoch == 0:
        best_avg_acc_epoch = epoch
        best_avg_acc = avg_acc
        if f'-o={args.optimizer}-lr={args.lr}' in train_method:
            args.best_avg_acc_model_path = join(args.model_dir, 
                                                f'm-best_acc{save_id}-{train_method}-r={args.replicate}-s={args.seed}.pt')
        else:
            args.best_avg_acc_model_path = join(args.model_dir, 
                                                f'm-best_acc{save_id}-{train_method}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}.pt')

        args.best_avg_acc_model_path = args.best_avg_acc_model_path.replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')
        torch.save(model.state_dict(), args.best_avg_acc_model_path)
    args.results_dict[f'best_acc{save_id}_epoch'].append(best_avg_acc_epoch)
    
#     if args.replicate == 0:
#         if (epoch + 1) % 10 == 0:
#             args.epoch_model_path = join(args.model_dir, 
#                                          f'm-e-tm={train_method}-a={args.arch}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}-e={epoch}.pt')
#             torch.save(model.state_dict(), args.epoch_model_path)
    
    try:
        if min_acc > best_robust_acc or epoch == 0:
            best_robust_acc_epoch = epoch
            best_robust_acc = min_acc
            
            if f'-o={args.optimizer}-lr={args.lr}' in train_method:
                args.best_robust_acc_model_path = join(args.model_dir, 
                                                       f'm-best_acc_r{save_id}-{train_method}-r={args.replicate}-s={args.seed}.pt')
            else:
                args.best_robust_acc_model_path = join(args.model_dir, 
                                                       f'm-best_acc_r{save_id}-{train_method}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}.pt')

            args.best_robust_acc_model_path = args.best_robust_acc_model_path.replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')
            torch.save(model.state_dict(), args.best_robust_acc_model_path)
        args.results_dict[f'best_robust_acc{save_id}_epoch'].append(best_robust_acc_epoch)
        
        return (best_loss, best_loss_epoch), (best_avg_acc, best_avg_acc_epoch), (best_robust_acc, best_robust_acc_epoch)
    except Exception as e:
        raise e
        return (best_loss, best_loss_epoch), (best_avg_acc, best_avg_acc_epoch), (None, None)
    
    
    
    
def log_data(dataset, header, indices=None):
    print(header)
    dataset_groups = dataset.targets_all['group_idx']
    if indices is not None:
        dataset_groups = dataset_groups[indices]
    groups = np.unique(dataset_groups)
    
    try:
        max_target_name_len = np.max([len(x) for x in dataset.class_names])
    except Exception as e:
        print(e)
        max_target_name_len = -1
    
    for group_idx in groups:
        counts = np.where(dataset_groups == group_idx)[0].shape[0]
        try:  # Arguably more pretty stdout
            group_name = dataset.group_labels[group_idx]
            group_name = group_name.split(',')
            group_name[0] += (' ' * int(
                np.max((0, max_target_name_len - len(group_name[0])))
            ))
            group_name = ','.join(group_name)
            print(f'- {group_name} : n = {counts}')
        except Exception as e:
            print(e)
            print(f'- {group_idx} : n = {counts}')