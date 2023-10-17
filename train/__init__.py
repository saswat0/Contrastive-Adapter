"""
Basic training and evaluation functions
"""
import os
import numpy as np
import torch

from tqdm import tqdm

from utils.logging import summarize_acc
from datasets import get_class_label_from_group_pred


def prep_metrics(dataloader, args):
    running_loss = 0.0
    correct = 0
    total = 0
    
    targets_s = dataloader.dataset.targets_all['spurious']
    targets_t = dataloader.dataset.targets_all['target']
    correct_by_groups = np.zeros([args.num_classes,
                                  args.num_classes])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)
    return running_loss, correct, total, targets_s, targets_t, correct_by_groups, total_by_groups, losses_by_groups
    


def train_model(model, optimizer, criterion, num_epochs,
                train_loader, val_loader, test_loader, 
                args, verbose=False):
    
    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []

    best_val_error = 1e10
    best_val_acc = 0
    
    for e in range(num_epochs):
        train_metrics = train_epoch(model, train_loader,
                                    optimizer, criterion, args)
        val_metrics = evaluate(model, val_loader, criterion, args)
        test_metrics = evaluate(model, test_loader,
                                criterion, args)
        
        running_loss, correct, total, correct_by_groups, total_by_groups = train_metrics
        if verbose:
            print(f'Train Epoch {e}')
        avg_acc, min_acc = summarize_acc(correct_by_groups, total_by_groups,
                                         stdout=verbose)
        print(f'Train Epoch {e} | loss: {running_loss:<.2f} | avg acc: {avg_acc:<.2f}% | robust acc: {min_acc:<.2f}%')
        all_train_loss.append(running_loss)
        all_train_acc.append(avg_acc)
        
        running_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
        if verbose:
            print(f'Val Epoch {e}')
        avg_acc, min_acc = summarize_acc
        avg_acc, min_acc = summarize_acc(correct_by_groups, total_by_groups,
                                         stdout=verbose)
        print(f'Val   Epoch {e} | loss: {running_loss:<.2f} | avg acc: {avg_acc:<.2f}% | robust acc: {min_acc:<.2f}%')
        all_val_loss.append(running_loss)
        all_val_acc.append(avg_acc)
            
        running_loss, correct, total, correct_by_groups, total_by_groups = test_metrics
        if verbose:
            print(f'Test Epoch {e}')
        avg_acc, min_acc = summarize_acc
        avg_acc, min_acc = summarize_acc(correct_by_groups, total_by_groups,
                                         stdout=verbose)
        print(f'Test  Epoch {e} | loss: {running_loss:<.2f} | avg acc: {avg_acc:<.2f}% | robust acc: {min_acc:<.2f}%')
        
    return model


def train_epoch(model, dataloader, optimizer, criterion, args, evaluate=True):
    running_loss, correct, total, targets_s, targets_t, correct_by_groups, total_by_groups, losses_by_groups = prep_metrics(dataloader, args)
    
    targets_s = dataloader.dataset.targets_all['spurious']
    targets_t = dataloader.dataset.targets_all['target']
    
    
    try:
        model.to_device(args.device)
    except:    
        model.to(args.device)
        
    model.train()
    if args.replicate == 20:
        model.train()
    elif args.replicate == 30:
        model.eval()
    model.zero_grad()
    
    pbar = enumerate(dataloader)
    for batch_ix, data in pbar:
        
        try:
            inputs, labels, data_ix = data
        except ValueError:
            inputs, labels, _, _, data_ix = data
            
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        labels_target = [targets_t[ix] for ix in data_ix]
        labels_spurious = [targets_s[ix] for ix in data_ix]
        
        # Classifier forward pass
        y_outputs = model(inputs)
        loss = criterion(y_outputs, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Save performance
        _, predicted = torch.max(y_outputs.data, 1)
        total += labels.size(0)
        all_correct = (predicted == labels)
        correct += all_correct.sum().item()
        running_loss += loss.item()
        
        try:
            if args.group_to_class_mapping is not None:
                # Apply mapping function
                predicted = get_class_label_from_group_pred(predicted.detach().cpu().numpy(),
                                                            mapping=args.group_to_class_mapping)
                all_correct = torch.tensor(predicted == labels.detach().cpu().numpy())
        except:
            pass
        
        # Save group-wise acc
        for ix, s in enumerate(labels_spurious):
            y = labels_target[ix]
            correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
            total_by_groups[int(y)][int(s)] += 1
        
        inputs = inputs.cpu()
        loss = loss.cpu()
        
        
    try:
        model.to_device(torch.device('cpu'))
    except:    
        model.cpu()
        
    return running_loss, correct, total, correct_by_groups, total_by_groups


def get_predictions(model, dataloader, split, args):
    prediction_dir = args.embeddings_dir
    prediction_fname = f'predictions_d={args.dataset}-s={split}-c={args.config}-m={args.load_base_model}.pt'
    prediction_path = os.path.join(prediction_dir, prediction_fname)
    if os.path.exists(prediction_path):
        predictions = torch.load(prediction_path)
        return torch.load(prediction_path)
    
    predictions = []
    model.to(args.device)
    model.eval()
    
    pbar = enumerate(tqdm(dataloader, 
                          desc=f'Getting predictions on device {args.device}',
                          leave=False))
    
    with torch.no_grad():
        for batch_ix, data in pbar:
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            # Save performance
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu())
    predictions = torch.cat(predictions)
    torch.save(predictions, prediction_path)
    return predictions
    
        
      
def evaluate(model, dataloader, criterion, args,
             save_embeddings=False):
    running_loss, correct, total, targets_s, targets_t, correct_by_groups, total_by_groups, losses_by_groups = prep_metrics(dataloader, args)
    
    targets_s = dataloader.dataset.targets_all['spurious']
    targets_t = dataloader.dataset.targets_all['target']
    
    try:
        model.to_device(args.device)
    except:    
        model.to(args.device)
    model.eval()
    
        
    pbar = enumerate(dataloader)
    
    data_embeddings = []
    
    with torch.no_grad():
        for batch_ix, data in pbar:
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            labels_target = [targets_t[dix] for dix in data_ix]
            labels_spurious = [targets_s[ix] for ix in data_ix]
            
            if save_embeddings and 'adapter' in args.train_method:
                outputs, embeddings = model(inputs, return_hidden=True)
                embeddings = embeddings.cpu()
                data_embeddings.append(embeddings)
            else:
                outputs = model(inputs)
                
            try:
                loss = criterion(outputs, labels)
            except Exception as e:
                print(outputs)
                raise e

            # Save performance
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_correct = (predicted == labels).detach().cpu()
            correct += all_correct.sum().item()
            running_loss += loss.item()
            
            try:
                if args.group_to_class_mapping is not None:
                    # Apply mapping function
                    predicted = get_class_label_from_group_pred(predicted.detach().cpu().numpy(),
                                                                mapping=args.group_to_class_mapping)
                    all_correct = torch.tensor(predicted == labels.detach().cpu().numpy())
            except:
                pass

            # Save group-wise accuracy
            labels = labels.detach().cpu().numpy()
            for ix, s in enumerate(labels_spurious):
                y = labels_target[ix]
                correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
                total_by_groups[int(y)][int(s)] += 1

            # Clear memory
            inputs = inputs.to(torch.device('cpu'))
            # labels = labels.to(torch.device('cpu'))  
            outputs = outputs.to(torch.device('cpu'))
            loss = loss.to(torch.device('cpu'))
            del outputs; del inputs; del labels; del loss
            
    try:
        model.to_device(torch.device('cpu'))
    except:    
        model.cpu()
        
    if save_embeddings:
        data_embeddings = torch.vstack(data_embeddings)
        return running_loss, correct, total, correct_by_groups, total_by_groups, data_embeddings
       
        
    return running_loss, correct, total, correct_by_groups, total_by_groups
