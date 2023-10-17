"""
Evaluation helper functions for pretrained models
"""

import os
from os.path import join

import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# CLIP
from clip import clip
# Pretrained models
from network import load_base_model
# Evaluation
from train.wilds import evaluate_wilds
from network.clip import evaluate_clip
from utils import initialize_save_paths
from utils.logging_wilds import summarize_acc, log_metrics, process_validation_metrics
from utils.logging import summarize_acc_from_predictions

# Dataset setup
from datasets import initialize_data, get_resampled_set

# Adapter and probe setup
from datasets.embeddings import get_embedding_loader
from network import get_adapter


# -------------
# Dataset Setup
# -------------

def setup_dataset_and_load_dataloaders(dataset_name, args, transform=None,
                                       return_visualizer=False):
    # Load data
    args.dataset = dataset_name
    load_dataloaders, visualize_dataset = initialize_data(args)
    args.config = f'config-'
    confounders = '_'.join(args.confounder_names)
    args.config += f'target={args.target_name}-confounders={confounders}'

    if not os.path.exists('../embeddings/'):
        os.makedirs('../embeddings/')
    args.embeddings_dir = join(f'../embeddings/{args.dataset}')
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
    args.embeddings_dir = join(args.embeddings_dir, args.config)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
    load_dataloaders, visualize_dataset = initialize_data(args)

    dataloaders_base = load_dataloaders(args, train_shuffle=False,
                                        transform=transform)

    if args.dataset in ['fmow']:
        train_loader_base, iid_val_loader_base, val_loader_base, iid_test_loader_base, test_loader_base = dataloaders_base
        splits = ['train', 'iid_val', 'val', 'iid_test', 'test']
    elif args.dataset in ['iwildcam']:
        train_loader_base, val_loader_base, test_loader_base, iid_val_loader_base, iid_test_loader_base = dataloaders_base
        splits = ['train', 'val', 'test', 'iid_val', 'iid_test']
    else:
        train_loader_base, val_loader_base, test_loader_base = dataloaders_base
        splits = ['train', 'val', 'test']
        
    if return_visualizer:
        return dataloaders_base, splits, visualize_dataset
    return dataloaders_base, splits


# --------------------
# Evaluation functions
# --------------------

def evaluate_waterbirds_predictions(predictions, dataloader):
    targets  = dataloader.dataset.targets_all['target']
    spurious = dataloader.dataset.targets_all['spurious']
    
    try:
        predictions = predictions.numpy()
    except:
        pass
    correct_by_group = [[0, 0], [0, 0]]
    total_by_group   = [[0, 0], [0, 0]]
    accs_by_group    = [[0, 0], [0, 0]]
    correct = predictions == targets
    for t in [0, 1]:
        for s in [0 ,1]:
            ix = np.where(np.logical_and(targets == t,
                                         spurious == s))[0]
            correct_by_group[t][s] += np.sum(correct[ix])
            total_by_group[t][s] += len(ix)
            accs_by_group[t][s] = np.sum(correct[ix]) / len(ix)
    
    # Average accuracy
    avg_acc = (
        correct_by_group[0][0] +
        correct_by_group[0][1] +
        correct_by_group[1][0] +
        correct_by_group[1][1]
    )
    avg_acc = avg_acc * 100 / np.sum(np.array(total_by_group))
    
    # Adjust average accuracy
    adj_avg_acc = (
        accs_by_group[0][0] * 3498 +
        accs_by_group[0][1] * 184 +
        accs_by_group[1][0] * 56 +
        accs_by_group[1][1] * 1057
    )
    adj_avg_acc = adj_avg_acc * 100 / (3498 + 184 + 56 + 1057)
    
    accs_by_group = np.array(accs_by_group).flatten() * 100
    
    worst_acc = np.min(accs_by_group)
    
    return worst_acc, adj_avg_acc, avg_acc, accs_by_group


def evaluate_dataset_prediction(predictions, dataloader, 
                                args, verbose=True):
    if args.dataset == 'celebA':
        try:
            predictions = predictions.cpu().numpy()
        except:
            pass
        avg_acc, min_acc = summarize_acc_from_predictions(
            predictions, dataloader, args, stdout=verbose
        )
    elif args.dataset == 'waterbirds':
        accs = evaluate_waterbirds_predictions(predictions, 
                                               dataloader)
        worst_acc, adj_avg_acc, avg_acc_, accs_by_group = accs
        avg_acc = adj_avg_acc
        min_acc = worst_acc
        if verbose:
            for ix, acc in enumerate(accs_by_group):
                print(f'Group {ix} acc: {acc:.2f}%')
            print(f'Worst-group acc: {worst_acc:.2f}%')
            print(f'Average acc:     {avg_acc_:.2f}%')
            print(f'Adj Average acc: {adj_avg_acc:.2f}%')
    else:
        try:
            predictions = torch.from_numpy(predictions)
        except:
            pass
        eval_dict, eval_str = evaluate_wilds(
            predictions, dataloader
        )
        avg_acc = eval_dict[args.average_group_key] * 100
        min_acc = eval_dict[args.worst_group_key] * 100
        if verbose:
            print(eval_str)
    return avg_acc, min_acc


# ----------------------------------
# Load model results and checkpoints
# ----------------------------------

def show_results(val_metric, test_robust_metric, test_avg_metric, root_dir,
                 filter_keywords,
                 max_epochs, print_files=False, topk=None):
    conditions = lambda x: [k in x for k in filter_keywords]  # _include and k not in filter_keywoards_exclude]
    configs = {}
    for f in os.listdir(root_dir):
        if '.csv' in f and np.sum(conditions(f)) == len(filter_keywords):
            if print_files:
                print(f)
            args = f.split('-')
            df = pd.read_csv(join(root_dir, f))
            max_epoch = df['epoch'].max()
            if max_epoch >= max_epochs - 1:
                m = f.split('-m=')[-1].split('-clip')[0]
                pm = f.split('-pm=')[-1].split('-t=')[0]

                try:
                    normalize_embeddings = f.split('-ne=')[-1].split('-')[0]
                except:
                    normalize_embeddings = ''
                backbone = f.split('clip_')[-1].split('_')[0]
                pred = f.split(backbone + '_')[-1].split('-bs_trn')[0]
                bs_trn = f.split('bs_trn=')[-1].split('-')[0]
                lr = f.split('-lr=')[-1].split('_')[0]
                wd = f.split('wd=')[-1].split('_mo')[0]
                cw = f.split('-cw=')[-1].split('-me')[0]
                if f.split('-r=')[-1].split('-')[-1][0] == 's':
                    r = f.split('-r=')[-1].split('-')[0]
                    s = f.split('-s=')[-1].split('.')[0]
                else:
                    r = f.split('-r=')[-1].split('.')[0]
                    s = f.split('-s=')[-1].split('-')[0]

                config = f'clip-{backbone}-m={m}-pred={pred}-ne={normalize_embeddings}-bs={bs_trn}-lr={lr}-wd={wd}-r={r}'
                test_robust_accs =  df['test_robust_acc']

                best_test_iloc = df[val_metric].argmax()
                # print(df['epoch'].iloc[best_test_iloc])
                best_test_robust_acc = df[test_robust_metric].iloc[best_test_iloc]

                if best_test_robust_acc > 0:
                    df = df[df[test_robust_metric] == best_test_robust_acc]
                    test_avg_acc = df[test_avg_metric].mean()
                    if config in configs:
                        try:
                            configs[config]['test_robust_acc'].append(best_test_robust_acc)
                            configs[config]['test_avg_acc'].append(test_avg_acc)
                            configs[config]['max_epoch'].append(max_epoch)
                            configs[config]['count'] += 1
                            configs[config]['file_paths'].append(join(root_dir, f))
                        except:
                            configs[config] = {'test_robust_acc': [best_test_robust_acc],
                                               'test_avg_acc': [test_avg_acc],
                                               'max_epoch': [max_epoch],
                                               'count': 1,
                                               'file_paths': [join(root_dir, f)]}
                    else:
                        configs[config] = {'test_robust_acc': [best_test_robust_acc],
                                           'test_avg_acc': [test_avg_acc],
                                           'max_epoch': [max_epoch],
                                           'count': 1,
                                           'file_paths': [join(root_dir, f)]}
    all_configs = []
    all_means = {'test_robust_acc': [], 'test_avg_acc': [], 'max_epoch': [], 'count': [], 'file_paths': []}
    all_stds = {'test_robust_acc': [], 'test_avg_acc': [], 'max_epoch': [], 'count': [], 'file_paths': []}
    for config in configs:
        model_results = configs[config]
        all_configs.append(config)
        for name in model_results:
            if name != 'file_paths':
                mean = np.mean(model_results[name])
                std  = np.std(model_results[name])
            else:
                mean = model_results[name]
                std = model_results[name]
            all_means[name].append(mean)
            all_stds[name].append(std)
    sort_ix = np.argsort(all_means['test_robust_acc'])[::-1]
    for ix in sort_ix[:topk]:
        print(all_configs[ix])
        for name in ['test_robust_acc', 'test_avg_acc', 'max_epoch']:
            print(f'{name:11s}: {all_means[name][ix]:>2.1f} ± {all_stds[name][ix]:.1f}%')
        print(f'count: {all_means["count"][ix]}')
        print('---')
    return [all_means['file_paths'][ix] for ix in sort_ix[:topk]]


def get_model_args_from_result_path(result_path, args):
    args.hidden_dim = int(result_path.split('-hd=')[-1].split('-')[0])
    args.projection_dim = int(result_path.split('-pd=')[-1].split('-')[0])
    args.input_dim = args.base_model_dim 
    args.adapter_head_batch_norm = bool(int(result_path.split('-ahbn=')[-1].split('-')[0]))
    args.temperature = float(result_path.split('-ct=')[-1].split('-')[0])
    try:
        args.num_encoder_layers = int(result_path.split('_nel=')[-1].split('-')[0])
    except:
        args.num_encoder_layers = 0
        
    if 'clip' in result_path:
        args.train_method = result_path.split('-m=')[-1].split('-clip')[0]
    elif 'Eleuther' in result_path:
        args.train_method = result_path.split('-m=')[-1].split('-Eleuther')[0] 
    elif 'cloob' in result_path:
        args.train_method = result_path.split('-m=')[-1].split('-cloob')[0]
        
    args.residual_connection = bool(int(result_path.split('-rc=')[-1].split('-')[0]))
    args.classification_temperature = float(result_path.split('-ct=')[-1].split('-')[0])
    if 'nel' in args.train_method:
        args.train_method = args.train_method.split('_nel')[0]
        
        
def get_model_path_from_result_path(result_path, args, metric='best_acc_r',
                                    model_root_dir='../models/',
                                    config_dir=None, 
                                    adapter_dir=None,  
                                    verbose=False):
    if config_dir is None:
        config_dir = args.config_dir
    model_dir = join(model_root_dir, f'{args.dataset}', config_dir, adapter_dir)
    if verbose:
        print(model_dir)
    model_components = result_path.replace('bs_trn', 'bs_tr').split('/')[-1].split('-')
    
    if '-op' in result_path:
        model_components.append('_op')
    else:
        pass

    ignore_ix = 0
    ignore = True
    for ix in range(len(model_components)):
        if 'm=' not in model_components[ix] and ignore:
            ignore_ix += 1
            ignore = True
        else:
            ignore = False
            
        if 'EleutherAI_' in model_components[ix]:
            model_components[ix] = model_components[ix].replace('EleutherAI_', '')
            
        if 'pclw' in model_components[ix]:
            model_components[ix] += '_'
            
        if 's=' in model_components[ix] and 'gs' not in model_components[ix]:
            model_components[ix] = '-' + model_components[ix]
            
        if 'r=' in model_components[ix] and 't' not in model_components[ix]:
            model_components[ix] = '-' + model_components[ix]

        if 'lr=' in model_components[ix] and '_wd' in model_components[ix]:
            if model_components[ix][-1] == 'e':
                model_components[ix] = model_components[ix] + '-' + model_components[ix + 1]
            model_components[ix + 1] = model_components[ix + 1].split('_')[-1]
            lr, wd, mo = model_components[ix].split('_')
            model_components[ix] = lr
            model_components.append(wd)
            model_components.append(mo)

        if '.csv' in model_components[ix]:
            model_components[ix] = model_components[ix].split('.')[0]

    model_components.remove('cm=supcon')
    model_components.append(f'm-{metric}-m=')

    all_model_paths = []
    all_offending_components = []
    for f in os.listdir(model_dir):
        select = True
        offending_components = []
        for c in model_components[ignore_ix:]:
            if c not in f:
                select = False
                offending_components.append(c)
        if select:
            all_model_paths.append(join(model_dir, f))
        all_offending_components.append(offending_components)
            
    if len(all_model_paths) == 1:
        if verbose:
            print(all_model_paths[0])
        return all_model_paths[0]
    elif len(all_model_paths) == 0:
        print(model_components[ignore_ix:])
        for ix, c in enumerate(all_offending_components):
            if len(c) < 2:
                
                print(os.listdir(model_dir)[ix])
                print(c)
        raise ValueError(f'Should have one model path selected')
    else:
        print(model_components[ignore_ix:])
        for model_path in all_model_paths:
            print(model_path)
            if '_op_' not in model_path and '-op' not in result_path:
                return model_path
        raise ValueError('Filtering critera not specific enough. Only one model path should be selected.')
        
        
def get_model_from_result_path(result_path, query_embeddings, args, 
                               metric='best_acc_r',
                               model_root_dir='../models/',
                               config_dir=None,  # args.config_dir,
                               verbose=False):
    get_model_args_from_result_path(result_path, args)
    if config_dir is None:
        config_dir = args.config_dir
    adapter_dir = args.adapter_dir  
    model_path = get_model_path_from_result_path(result_path, args, metric,
                                                 model_root_dir, config_dir,
                                                 adapter_dir, verbose)
    model = get_adapter(args.num_classes, args, query_embeddings)
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(model_path)
        print(args.base_model_dim)
        raise e
    return model   


# -------------------
# Adapter predictions
# -------------------

def get_adapter_embeddings(adapter, embedding_loader, device):
    adapter.to(device)
    adapter.eval()
    all_embeddings = []
    with torch.no_grad():
        for ix, data in enumerate(tqdm(embedding_loader, 
                                       desc=f'Computing adapter output embeddings',
                                       leave=False)):
            inputs, labels, data_ix = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            embeddings = adapter.encode(inputs).cpu() 
            all_embeddings.append(embeddings)
            inputs = inputs.cpu()
            labels = labels.cpu()
    adapter.cpu()
    return torch.vstack(all_embeddings)


def get_adapter_predictions(adapter, embedding_loader, device):
    try:
        adapter.to_device(device)
    except:
        adapter.to(device)
    adapter.eval()
    all_predictions = []
    with torch.no_grad():
        for ix, data in enumerate(embedding_loader):
            inputs, labels, data_ix = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            try:
                outputs = adapter.forward(inputs, return_hidden=False).cpu()
            except:
                outputs = adapter(inputs).cpu()
            _, predictions = torch.max(outputs.data, 1)
            all_predictions.append(predictions)
            inputs = inputs.cpu()
            labels = labels.cpu()
    adapter.cpu()
    return torch.cat(all_predictions)


def get_adapter_predictions_from_embeddings(query_embeddings,
                                            adapter_embeddings,
                                            temperature):
    q = query_embeddings / query_embeddings.norm(dim=-1, keepdim=True)
    k = adapter_embeddings / adapter_embeddings.norm(dim=-1, keepdim=True)
    probs = (temperature * (k @ q.T)).softmax(dim=-1)
    _, predictions = torch.max(probs.data, 1)
    return predictions


def get_prediction_logits_from_embeddings(class_embeddings,
                                          sample_embeddings,
                                          temperature):
    q = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    k = sample_embeddings / sample_embeddings.norm(dim=-1, keepdim=True)
    return temperature * (k @ q.T)


# -----------------------------------------------
# Linear probe + ensembling (WiSE-FT) predictions
# -----------------------------------------------

class ClassificationHead(nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)
    
    
def weight_space_ensemble(model_a, model_b, alpha):
    theta_a = {k: v.clone() for k, v in model_a.state_dict().items()}
    theta_b = {k: v.clone() for k, v in model_b.state_dict().items()}
    theta = {
        key: (1 - alpha) * theta_a[key] + alpha * theta_b[key]
        for key in theta_a.keys()
    }
    model = copy.deepcopy(model_a)
    model.load_state_dict(theta)
    return model

def weight_space_average(models):
    thetas = []
    for model in models:
        thetas.append({k: v.clone() for k, v in model.state_dict().items()})
    theta = {key: torch.stack([_theta[key] for _theta in thetas]).mean(dim=0) 
             for key in thetas[0].keys()}
    model = copy.deepcopy(models[0])
    model.load_state_dict(theta)
    return model


def get_prediction_logits_from_embeddings(class_embeddings,
                                          sample_embeddings,
                                          temperature):
    q = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    k = sample_embeddings / sample_embeddings.norm(dim=-1, keepdim=True)
    return temperature * (k @ q.T)


def get_prediction_logits_from_classification_head(classifier,
                                                   embedding_loader, 
                                                   device):
    classifier.to(device)
    all_logits = []
    with torch.no_grad():
        for ix, data in enumerate(embedding_loader):
            inputs, labels, data_ix = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = classifier(inputs).cpu()
            all_logits.append(outputs)
            inputs = inputs.cpu()
            labels = labels.cpu()
    classifier.cpu()
    return torch.vstack(all_logits)


def get_wise_ft_predictions_from_logits(logits_ft, logits_base, alpha):
    probs = ((1 - alpha) * logits_ft + alpha * logits_base).softmax(dim=-1)
    _, predictions = torch.max(probs.data, 1)
    return predictions


def get_output_ensemble_accuracies(classifier, zeroshot_logits,
                                   embedding_loader, base_dataloader,
                                   device, alphas, args, verbose=False):
    classifier_logits = get_prediction_logits_from_classification_head(
        classifier, embedding_loader, device
    )
    average_accs = []
    robust_accs  = []
    for alpha in alphas:
        predictions = get_wise_ft_predictions_from_logits(
            classifier_logits, zeroshot_logits, alpha
        )
        avg_acc, wg_acc = evaluate_dataset_prediction(predictions,
                                                      base_dataloader,
                                                      args, verbose=verbose)
        average_accs.append(avg_acc)
        robust_accs.append(wg_acc)
    return average_accs, robust_accs, alphas


def get_output_model_average_accuracies(classifier, query_embeddings,
                                        embedding_loader, base_dataloader,
                                        device, alphas, args, verbose=False):
    zeroshot_weights = query_embeddings * 100.
    zeroshot_cls = ClassificationHead(normalize=True, 
                                      weights=zeroshot_weights)
    linear_probe_params = []
    for p in classifier.parameters():
        linear_probe_params.append(p.data)
    linear_cls = ClassificationHead(normalize=False,
                                    weights=linear_probe_params[0],
                                    biases=linear_probe_params[1])
    average_accs = []
    robust_accs  = []
    for alpha in alphas:
        wise_ft_cls = weight_space_ensemble(linear_cls, zeroshot_cls, alpha)
        predictions = get_adapter_predictions(wise_ft_cls,
                                              embedding_loader,
                                              args.device)
        avg_acc, wg_acc = evaluate_dataset_prediction(predictions,
                                                      base_dataloader,
                                                      args, verbose=verbose)
        average_accs.append(avg_acc)
        robust_accs.append(wg_acc)
    return average_accs, robust_accs, alphas


def get_prediction_logits(adapter, embedding_loader, device):
    try:
        adapter.to_device(device)
    except:
        adapter.to(device)
    adapter.eval()
    all_predictions = []
    with torch.no_grad():
        for ix, data in enumerate(embedding_loader):
            inputs, labels, data_ix = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            try:
                outputs = adapter.forward(inputs, return_hidden=False).cpu()
            except:
                outputs = adapter(inputs).cpu()
            _, predictions = torch.max(outputs.data, 1)
            all_predictions.append(predictions)
            inputs = inputs.cpu()
            labels = labels.cpu()
    adapter.cpu()
    return torch.cat(all_predictions)


def get_ensemble_outputs(logits_zeroshot, alphas, 
                         embedding_loader, base_dataloader, 
                         model_path, args, verbose=True,
                         metric='best_acc_r'):
    model_ = get_model_from_result_path(model_path,
                                        args.query_embeddings, 
                                        args, 
                                        metric=metric,
                                        verbose=verbose)
    predictions = get_adapter_predictions(model_, 
                                          embedding_loader,
                                          args.device)
    avg_acc, wg_acc = evaluate_dataset_prediction(predictions,
                                                  base_dataloader,
                                                  args, verbose=verbose)
    if '=linear' in model_path:
        ensemble_outputs = get_output_model_average_accuracies(
            model_, args.query_embeddings, 
            embedding_loader, base_dataloader,
            args.device, alphas, args, verbose=verbose)
    else:
        ensemble_outputs = get_output_ensemble_accuracies(
            model_, logits_zeroshot, embedding_loader, base_dataloader,
            args.device, alphas, args, verbose=verbose
        )
    if verbose:
        print(avg_acc, wg_acc)
    return ensemble_outputs


# --------
# Plotting
# --------

def get_base_model_title(base_model):
    title = ''
    backbone_map = {'RN50':   'ResNet-50',
                    'RN101':  'ResNet-101',
                    'ViTB32': 'ViT-B/32',
                    'ViTB16': 'ViT-B/16',
                    'ViTL14': 'ViT-L/14'}
    parts = base_model.split('_')
    if parts[0] == 'clip':
        title += f'CLIP {backbone_map[parts[1]]}'
    else:
        title += base_model
        
    return title


def adjust_plot(ax, xrange, yrange, tick_freq=0.05, log_adjust=True):
    
    ax.grid(linewidth=0.5)

    if type(tick_freq) is tuple:
        xtick_freq, ytick_freq = tick_freq[0], tick_freq[1]
    else:
        xtick_freq, ytick_freq = tick_freq, tick_freq

    if log_adjust:
        h = lambda p: np.log(p / (1 - p))
    else:
        h = lambda p: p

    def transform(z):
        return np.vectorize(h)(z)  # [h(p) for p in z]
    

    tick_loc_x = np.array([round(z, 2) for z in 
                           np.arange(xrange[0], xrange[1], xtick_freq)])
    ax.set_xticks(transform(tick_loc_x))
    ax.set_xticklabels([str(round(loc * 100)) for loc in tick_loc_x], fontsize=13)

    tick_loc_y = np.array([round(z, 2) for z 
                           in np.arange(yrange[0], yrange[1], ytick_freq)])
    ax.set_yticks(transform(tick_loc_y))
    ax.set_yticklabels([str(round(loc * 100)) for loc in tick_loc_y], fontsize=13)

    z = np.arange(min(xrange[0], yrange[0]), max(xrange[1], yrange[1]), 0.01)
    
    ax.set_ylim(h(yrange[0]) * 0.5, h(yrange[1]) * 1.05)
    ax.set_xlim(h(xrange[0]) * 0.95, h(xrange[1]) * 1.05)

    

    return transform


def plot_ensemble(ax, ensemble_outputs_x, ensemble_outputs_y, alphas, 
                  transform, zeroshot_ix, trained_ix, ensemble_ix,
                  color, 
                  zeroshot_label='CLIP Zero-Shot',
                  trained_label='Nonlinear Probe',
                  ensemble_label='WiSE-FT',
                  plot_ensemble=True,
                  plot_zeroshot=True,
                  ensemble_marker_size=50,
                  single_model_marker_size=200,
                  percentage=True,
                  zorder=1):
    ensemble_outputs_x = np.array(ensemble_outputs_x)
    ensemble_outputs_y = np.array(ensemble_outputs_y)
    if percentage:
        ensemble_outputs_x = ensemble_outputs_x  / 100.
        ensemble_outputs_y = ensemble_outputs_y  / 100.
    
    if plot_ensemble:
        ax.plot(transform(ensemble_outputs_x), 
                transform(ensemble_outputs_y),
                alpha=0.5, c=color, zorder=zorder)
    
    if plot_zeroshot:
        ax.scatter(
            transform(ensemble_outputs_x[zeroshot_ix]), 
            transform(ensemble_outputs_y[zeroshot_ix]),
            label=zeroshot_label, marker='*', 
            s=single_model_marker_size, alpha=0.8, 
            c='white', edgecolors='black', zorder=zorder+2
        )
    ax.scatter(
        transform(ensemble_outputs_x[trained_ix]), 
        transform(ensemble_outputs_y[trained_ix]),
        label=trained_label, marker='*', 
        s=single_model_marker_size, alpha=0.8, 
        c=color, edgecolors='black', zorder=zorder+2
    )
    if plot_ensemble:
        ax.scatter(
            transform(ensemble_outputs_x[ensemble_ix]), 
            transform(ensemble_outputs_y[ensemble_ix]),
            label=ensemble_label, marker='d', 
            s=ensemble_marker_size, alpha=0.6, c=color,
            zorder=zorder
        )
        
        
def plot_outputs(ax, all_ensemble_outputs, dataset, 
                 log_adjust=False, average_x_axis=True,
                 plot_adapter_ensemble=False):

    if average_x_axis:
        avg_ax = 0
        min_ax = 1        
    else:
        avg_ax = 1
        min_ax = 0
        
    ensemble_outputs_x = np.concatenate(
        [outputs[avg_ax] for outputs in all_ensemble_outputs]) / 100.
    ensemble_outputs_y = np.concatenate(
        [outputs[min_ax] for outputs in all_ensemble_outputs]) / 100.
        
    x_range = (min(ensemble_outputs_x), max(ensemble_outputs_x))
    y_range = (min(ensemble_outputs_y), max(ensemble_outputs_y))
        
    transform = adjust_plot(ax, x_range, y_range, log_adjust=log_adjust)
    plot_ensemble(ax, 
                  all_ensemble_outputs[2][avg_ax],
                  all_ensemble_outputs[2][min_ax],
                  alphas, transform,
                  zeroshot_ix, trained_ix, wise_ft_ix,
                  color='tab:blue',
                  zeroshot_label='CLIP Zero-Shot',
                  trained_label='Adapter',
                  ensemble_label='WiSE-FT Adapter',
                  plot_ensemble=plot_adapter_ensemble,
                  plot_zeroshot=False,
                  ensemble_marker_size=50,
                  single_model_marker_size=300,
                  percentage=True,
                  zorder=2)
    
    plot_ensemble(ax, 
                  all_ensemble_outputs[0][avg_ax],
                  all_ensemble_outputs[0][min_ax], 
                  alphas, transform,
                  zeroshot_ix, trained_ix, wise_ft_ix,
                  color='tab:green',
                  zeroshot_label='CLIP Zero-Shot',
                  trained_label='Linear Probe',
                  ensemble_label='WiSE-FT Linear',
                  plot_ensemble=True,
                  plot_zeroshot=True,
                  ensemble_marker_size=50,
                  single_model_marker_size=300,
                  percentage=True)

    plot_ensemble(ax, 
                  all_ensemble_outputs[1][avg_ax],
                  all_ensemble_outputs[1][min_ax],
                  alphas, transform,
                  zeroshot_ix, trained_ix, wise_ft_ix,
                  color='tab:orange',
                  zeroshot_label='CLIP Zero-Shot',
                  trained_label='Nonlinear Probe',
                  ensemble_label='WiSE-FT Nonlinear',
                  plot_ensemble=True,
                  plot_zeroshot=False,
                  ensemble_marker_size=50,
                  single_model_marker_size=300,
                  percentage=True)

    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.18), 
        ncol=3, fancybox=True, labelspacing=1
    )
    if average_x_axis:
        ax.set_xlabel('Average Accuracy %', fontsize=16)
        ax.set_ylabel(f'Worst-group Accuracy %', fontsize=16)
    else:
        ax.set_xlabel(f'Worst-group Accuracy %', fontsize=16)
        ax.set_ylabel('Average Accuracy %', fontsize=16)
        
    ax.set_title(f'{dataset} ({get_base_model_title(args.load_base_model)})', fontsize=16)
    
    
# -----------------------------
# Combined / pipeline functions
# -----------------------------

def get_query_embeddings_and_embedding_loaders(dataloaders_base, splits, args):
    base_model_args = args.load_base_model.split('_')
    base_model_components = load_base_model(base_model_args, args, clip)
    base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions = base_model_components

    query_embeddings = get_embeddings(args.text_descriptions,
                                      base_model,
                                      args,
                                      normalize=True,
                                      verbose=False)
    
    # Get pretrained model dataset embeddings
    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = get_dataset_embeddings(
            base_model, dataloaders_base, args, split=split)
        
    embedding_loaders = [
        get_embedding_loader(dataset_embeddings[splits[idx]],
                             dataloader.dataset,
                             shuffle=False,
                             args=args,
                             batch_size=128)
        for idx, dataloader in enumerate(dataloaders_base)
    ]
    return query_embeddings, dataset_embeddings, embedding_loaders


def get_ensemble_outputs_from_model_paths(model_paths,
                                          logits_zeroshot,
                                          alphas, 
                                          embedding_loader, 
                                          base_dataloader,
                                          args, verbose,
                                          multiple_seeds=False,
                                          metric='best_acc_r'):
    outputs = [[], []]
    if multiple_seeds:
        for model_path_seeds in model_paths:
            # Hardcoded, but avg, robust, alpha
            all_ensemble_outputs = [[], [], []]
            for model_path in model_path_seeds:
                ensemble_outputs = get_ensemble_outputs(logits_zeroshot, 
                                                        alphas, 
                                                        embedding_loader, 
                                                        base_dataloader, 
                                                        model_path, args,
                                                        verbose,
                                                        metric=metric)
                ensemble_outputs = [np.array(o) for o in ensemble_outputs]
                for i, output in enumerate(ensemble_outputs):
                    all_ensemble_outputs[i].append(output)
            mean_ensemble_outputs = [np.mean(o, axis=0) 
                                     for o in all_ensemble_outputs]
            stdv_ensemble_outputs = [np.std(o, axis=0)
                                     for o in all_ensemble_outputs]
            outputs[0].append(mean_ensemble_outputs)
            outputs[1].append(stdv_ensemble_outputs)
    else:
        for model_path in model_paths:
            ensemble_outputs = get_ensemble_outputs(logits_zeroshot, 
                                                    alphas, 
                                                    embedding_loader, 
                                                    base_dataloader, 
                                                    model_path, args,
                                                    verbose,
                                                    metric=metric)
            # means
            outputs[0].append(ensemble_outputs)
            # stdevs
            outputs[1].append(None)
    return outputs


# ---------------
# Display results
# ---------------
def get_metrics(output_means, output_stdvs, method_ix):
     # Average
    avg_mean = output_means[0][method_ix]
    avg_stdv = output_stdvs[0][method_ix]
    # Worst-group
    wg_mean  = output_means[1][method_ix]
    wg_stdv  = output_stdvs[1][method_ix]
    return (wg_mean, wg_stdv), (avg_mean, avg_stdv)


def add_results(metrics, decimals):
    results = []
    for metric in metrics:
        results.append(
            f'\${metric[0]:.{decimals}f} \\pm {metric[1]:.{decimals}f}\$')
    return results


def display_results(ensemble_output_means, 
                    ensemble_output_stdvs,
                    zeroshot_name='CLIP Zero-Shot',
                    zeroshot_ix=-1,
                    trained_ix=0,
                    decimals=1,
                    print_text=False):
    results_dict = {'Accuracy': ['Worst-group', 'Average']}
    method_names = ['Linear Probe', 'Nonlinear Probe', 'Adapter']
    
    for _ix in range(len(method_names)):
        output_means = ensemble_output_means[_ix]
        output_stdvs = ensemble_output_stdvs[_ix]

        max_avg_acc_ix = np.argmax(output_means[0])
        max_min_acc_ix = np.argmax(output_means[1])

        # Zero-shot
        if _ix == 0:
            metrics = get_metrics(output_means, 
                                  output_stdvs, 
                                  zeroshot_ix)
            results_dict[zeroshot_name] = add_results(metrics, decimals)

        # Trained
        metrics = get_metrics(output_means, output_stdvs, trained_ix)
        results_dict[method_names[_ix]] = add_results(metrics, decimals)
        metrics = get_metrics(output_means, output_stdvs, max_min_acc_ix)
        results_dict[f'WiSE ({method_names[_ix]})'] = add_results(metrics, decimals)
    
    df = pd.DataFrame(results_dict)
    if print_text:
        print(df.to_string(index=False))
    else:
        display(df.style.hide_index())
    
    
def norm_embeddings(e):
    return e / e.norm(dim=1, keepdim=True)


def get_worst_mean_cos_sims(sample_embeddings, query_embeddings,
                            targets_t, targets_s, targets_g,
                            ratio=0.1):
    embeddings_by_class = []
    for t in np.unique(targets_t):
        class_indices = np.where(targets_t == t)[0]
        embeddings_same_class_by_group = {}
        for g in np.unique(targets_g):
            group_indices = np.where(np.logical_and(
                targets_t == t,
                targets_g == g
            ))[0]
            if len(group_indices) > 0:
                embeddings_same_class_by_group[g] = sample_embeddings[group_indices]

        embeddings_by_class.append(embeddings_same_class_by_group)
        
    all_mean_cos_sims = {}

    for class_id in np.unique(targets_t):
        average_sims = {}
        for g in embeddings_by_class[class_id]:
            cos_sims = (
                norm_embeddings(embeddings_by_class[class_id][g]) @ 
                norm_embeddings(query_embeddings).T
            )
            
            _, nearest_class_ids = cos_sims.topk(2, dim=1)
            
            _class_ids = []
            for ix in nearest_class_ids:
                if ix[0] == class_id:
                    _class_ids.append(ix[1])
                else:
                    _class_ids.append(ix[0])
            
            _class_ids = np.array(_class_ids)
            
            
            average_sims[g] = (cos_sims[:, class_id] - 
                               cos_sims[:, _class_ids])

            worst_size = int(np.round(len(average_sims[g]) * ratio))
            worst_sims, worst_ix = torch.abs(average_sims[g]).topk(
                worst_size, largest=False)

            all_mean_cos_sims[f'c={class_id}_g={g}'] = worst_sims.mean().item()
    return all_mean_cos_sims, [v for k, v in all_mean_cos_sims.items()]


def get_worst_mean_cos_sims_groupwise(sample_embeddings, query_embeddings,
                            targets_t, targets_s, targets_g,
                            ratio=0.1):
    embeddings_by_class = []
    for t in np.unique(targets_t):
        class_indices = np.where(targets_t == t)[0]
        embeddings_same_class_by_group = {}
        for g in np.unique(targets_g):
            group_indices = np.where(np.logical_and(
                targets_t == t,
                targets_g == g
            ))[0]
            if len(group_indices) > 0:
                embeddings_same_class_by_group[g] = sample_embeddings[group_indices]

        embeddings_by_class.append(embeddings_same_class_by_group)
        
    all_cos_sims = {}

    for class_id in np.unique(targets_t):
        cos_sims_by_class = []
        average_sims = {}
        for ix, g in embeddings_by_class[class_id].items():
            for ix_, g_ in embeddings_by_class[class_id].items():
                if ix != ix_:
                    shape_ = g.shape
                    _shape_ = g_.shape
                    cos_sims = (
                        norm_embeddings(g) @ 
                        norm_embeddings(g_).T
                    ).numpy()
                    cos_sims_by_class.append(cos_sims)
        all_cos_sims[class_id] = cos_sims_by_class
    return all_cos_sims



def evaluate_group_dataset_predictions(predictions, dataloader,
                                       args, verbose=True):
    if args.dataset == 'celebA':
        try:
            predictions = predictions.cpu().numpy()
        except:
            pass
        avg_acc, min_acc, accs_by_group = summarize_acc_from_predictions(
            predictions, dataloader, args, stdout=verbose, return_groups=True
        )
        return avg_acc, min_acc, accs_by_group
    elif args.dataset == 'waterbirds':
        accs = evaluate_waterbirds_predictions(predictions, 
                                               dataloader)
        worst_acc, adj_avg_acc, avg_acc_, accs_by_group = accs
        avg_acc = adj_avg_acc
        min_acc = worst_acc
        if verbose:
            for ix, acc in enumerate(accs_by_group):
                print(f'Group {ix} acc: {acc:.2f}%')
            print(f'Worst-group acc: {worst_acc:.2f}%')
            print(f'Average acc:     {avg_acc_:.2f}%')
            print(f'Adj Average acc: {adj_avg_acc:.2f}%')
            
        return avg_acc, min_acc, accs_by_group
    else:
        try:
            predictions = torch.from_numpy(predictions)
        except:
            pass
        eval_dict, eval_str = evaluate_wilds(
            predictions, dataloader
        )
        avg_acc = eval_dict[args.average_group_key] * 100
        min_acc = eval_dict[args.worst_group_key] * 100
        if verbose:
            print(eval_str)
        return avg_acc, min_acc, eval_dict
    return avg_acc, min_acc



def get_embedding_metrics(sample_embeddings, query_embeddings,
                          eval_loader, args, epoch,
                          dict_cos_sims_metrics, dict_cos_sims_group_wise):
    eval_set = eval_loader.dataset
    targets_t = eval_set.targets_all['target']
    targets_s = eval_set.targets_all['spurious']
    targets_g = eval_set.targets_all['group_idx']
    
    all_mean_cos_sims, all_mean_cos_sims_list = get_worst_mean_cos_sims(sample_embeddings, query_embeddings,
                        targets_t, targets_s, targets_g,
                        ratio=1)
    
    predictions = get_adapter_predictions_from_embeddings(
        query_embeddings,
        sample_embeddings,
        temperature=100.
    )
    
    avg_acc, min_acc, eval_extra = evaluate_group_dataset_predictions(
        predictions, eval_loader, 
        args, verbose=False)
    if args.wilds_dataset:
        group_accs = [eval_extra[k] * 100 
                      for k in eval_extra if 'acc_subclass=' in k]
    else:
        if args.dataset == 'celebA':
            eval_extra = eval_extra.flatten()
        group_accs = eval_extra

    for k, v in enumerate(all_mean_cos_sims_list):
        dict_cos_sims_metrics['group'].append(k)
        dict_cos_sims_metrics['epoch'].append(epoch)
        dict_cos_sims_metrics['cos_sim_delta'].append(v)
        dict_cos_sims_metrics['group_acc'].append(group_accs[k])
        
        
    all_cos_sims_group_wise = get_worst_mean_cos_sims_groupwise(
        sample_embeddings, 
        query_embeddings, 
        targets_t, targets_s, targets_g, ratio=1
    )
    
    for k, v in all_cos_sims_group_wise.items():
        for group_sims in v:
            dict_cos_sims_group_wise['class'].append(k)
            dict_cos_sims_group_wise['epoch'].append(epoch)
            dict_cos_sims_group_wise['cos_sim'].append(group_sims.mean())
        
    
  
  