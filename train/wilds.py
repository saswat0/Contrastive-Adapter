"""
Training and evaluation functions for WILDS datasets

"""
import numpy as np
import torch
from torch.nn import CosineEmbeddingLoss

from tqdm import tqdm

from utils.logging import summarize_acc
from train import get_predictions as _get_predictions


def evaluate_wilds(predictions, dataloader, targets=None):
    try:
        try:
            group_indices = dataloader.dataset.source_dataset.group_indices
        except:
            group_indices = dataloader.dataset.group_indices
        eval_dict, eval_str = dataloader.dataset.eval(predictions, 
                                                  dataloader.dataset.y_array, 
                                                  dataloader.dataset.metadata_array,
                                                     valid_group_indices=group_indices)
    except:
        eval_dict, eval_str = dataloader.dataset.eval(predictions, 
                                                      dataloader.dataset.y_array, 
                                                      dataloader.dataset.metadata_array)
    return eval_dict, eval_str


def get_predictions(model, dataloader, args):
    return _get_predictions(model, dataloader, args)


def train_epoch(model, dataloader, optimizer, criterion, args,
                evaluate=True):
    """
    Train function called during each epoch for WILDS datasets
    """
    running_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    
    try:
        model.to_device(args.device)
    except:    
        model.to(args.device)
    model.train()
    optimizer.zero_grad()
    
    if len(dataloader) > 200:
        pbar = tqdm(enumerate(dataloader), leave=False)
    else:
        pbar = enumerate(dataloader)
    for batch_ix, data in pbar:  
        inputs, labels, metadata = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        # Classifier forward pass
        y_outputs = model(inputs)
        loss = criterion(y_outputs, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Save performance
        _, predicted = torch.max(y_outputs.data, 1)
        predicted = predicted.cpu()
        all_predictions.append(predicted)
        
        labels = labels.cpu()
        total += labels.size(0)

        all_correct = (predicted == labels)
        correct += all_correct.sum().item()
        running_loss += loss.item()
        
        inputs = inputs.cpu()
        loss = loss.cpu()
        
        if len(dataloader) > 200:
            pbar.set_description(
                f'Batch Idx: ({batch_ix}/{len(dataloader)}) | Loss: {running_loss / (batch_ix + 1):.3f} | Avg Acc: {correct / total * 100:.1f}% ({correct}/{total})')
    try:
        model.to_device(torch.device('cpu'))
    except:    
        model.cpu()
    
    if evaluate:
        try:
            if args.unlucky_leftover is True and len(torch.cat(all_predictions)) != len(dataloader.dataset):
                all_predictions.append(all_predictions[-1][0].unsqueeze(0))
            all_predictions = torch.cat(all_predictions)
        except:
            all_predictions = torch.cat(all_predictions)

        eval_dict, eval_str = evaluate_wilds(all_predictions, dataloader)
    else:
        eval_dict = None
        eval_str = None
    return running_loss, correct, total, eval_dict, eval_str


def evaluate(model, dataloader, criterion, args):
    running_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    
    try:
        model.to_device(args.device)
    except:    
        model.to(args.device)
    model.eval()
    
    with torch.no_grad():
        for batch_ix, data in enumerate(dataloader):
            inputs, labels, metadata = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            try:
                loss = criterion(outputs, labels)
            except Exception as e:
                print(outputs)
                raise e

            # Save performance
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            all_predictions.append(predicted)

            labels = labels.cpu()
            total += labels.size(0)
            
            all_correct = (predicted == labels)
            correct += all_correct.sum().item()
            running_loss += loss.item()

            # Clear memory
            inputs = inputs.cpu()
            outputs = outputs.cpu()
            loss = loss.cpu()
            del outputs; del inputs; del labels; del loss
            
    try:
        model.to_device(torch.device('cpu'))
    except:    
        model.cpu()
    
    try:
        if args.unlucky_leftover is True and len(torch.cat(all_predictions)) != len(dataloader.dataset):
            all_predictions.append(all_predictions[-1][0].unsqueeze(0))
        all_predictions = torch.cat(all_predictions)
    except:
        all_predictions = torch.cat(all_predictions)
        
    eval_dict, eval_str = evaluate_wilds(all_predictions, dataloader)
    return running_loss, correct, total, eval_dict, eval_str

