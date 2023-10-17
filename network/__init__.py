import numpy as np
import torch
import torch.nn as nn

import copy
import json

from network.adapter import LinearProbe, MultiLayerAdapter

# Pretrained models
import network.clip as base_clip
import network.language_model as base_lm
from network.cloob import load_cloob_model


def get_optimizer(model, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, 
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, 
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)
    return optimizer


def get_adapter(num_classes, args, query_embeddings=None):
    if 'adapter_mt' in args.train_method:
        try:
            activation = args.activation
        except:
            activation = 'relu'
        softmax_output = True if 'softmax' in args.train_method else False
        model = MultiLayerAdapter(input_dim=args.base_model_dim,
                                  hidden_dim=args.hidden_dim,
                                  batch_norm=args.adapter_head_batch_norm,
                                  residual_connection=args.residual_connection,
                                  queries=query_embeddings,
                                  temperature=args.classification_temperature,
                                  return_hidden_by_default=False,
                                  num_encoder_layers=args.num_encoder_layers,
                                  activation=activation,
                                  softmax_output=softmax_output)
        
    elif args.train_method == 'linear_probe':
        model = LinearProbe(input_dim=args.base_model_dim,
                            num_classes=args.num_classes)

    else:
        raise NotImplementedError
    return model


def get_parameter_count(model, verbose=True):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    if verbose:
        print(f'-> Number of parameters: {num_params}')
    return num_params


def load_base_model(base_model_args, args, clip=None):
    """
    Load foundation model, foundation model transform, embedding functions
    Returns:
    - base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions
    """
    if 'clip' in base_model_args:
        clip_name = base_model_args[1]  # example: 'RN50'
        base_model, transform = clip.load(clip_name)
        args.num_base_model_parameters = get_parameter_count(base_model, 
                                                             verbose=True)
        pipeline = None
        base_transform = transform
        get_embeddings = base_clip.get_embeddings
        get_dataset_embeddings = base_clip.get_dataset_embeddings
        get_zeroshot_predictions = base_clip.get_zeroshot_predictions
        
    elif 'cloob' in base_model_args:
        base_model, transform = load_cloob_model(args)
        args.num_base_model_parameters = get_parameter_count(base_model, 
                                                             verbose=True)
        pipeline = None
        base_transform = transform
        get_embeddings = base_clip.get_embeddings
        get_dataset_embeddings = base_clip.get_dataset_embeddings
        get_zeroshot_predictions = base_clip.get_zeroshot_predictions
    
        
    else:
        # Ex.) --load_base_model 'EleutherAI/gpt-neo-1.3B_cls'
        if 'cls' in base_model_args:
            args.sequence_classification_model = True
            
        base_model, transform, tokenizer = base_lm.load_pretrained_language_model(
            args.sequence_classification_model, args
        )
        args.num_base_model_parameters = get_parameter_count(base_model, 
                                                             verbose=True)
        # Only use base model for feature extraction for now
        device_id = (torch.cuda.current_device() 
                     if torch.cuda.is_available() and not args.no_cuda else -1)
        pipeline = base_lm.load_pipeline(base_model, tokenizer,
                                         args.max_token_length,
                                         device=device_id,
                                         task='feature-extraction')
        base_model = pipeline
        base_transform = None
        get_embeddings = base_lm.get_embeddings
        get_dataset_embeddings = base_lm.get_dataset_embeddings
        get_zeroshot_predictions = base_lm.get_zeroshot_predictions
    return base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions


def freeze_weights(net):
    for p in net.parameters():
        p.requires_grad = False
    return net


def get_predictions(model, dataloader, args):
    predictions = []
    model.to(args.device)
    model.eval()
    
    pbar = tqdm(enumerate(dataloader))
    
    with torch.no_grad():
        for batch_ix, data in pbar:
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            # Save performance
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu())
    predictions = torch.cat(predictions)
    return predictions



"""
Model attributes, from https://github.com/kohpangwei/group_DRO/blob/master/models.py

Used for: Waterbirds
"""

model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'resnet18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    },  # CLIP input resolutions
    'RN50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'RN18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'RN101': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'RN50x16': {
        'feature_type': 'image',
        'target_resolution': (384, 384),
        'flatten': False
    },
    'ViTB32': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'ViTB16': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    }
}