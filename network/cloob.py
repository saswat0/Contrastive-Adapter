import os
import copy
import json
from os.path import join

import torch

from clip.clip import _transform
from clip.model import CLIPGeneral


def load_cloob_model(args,
                     model_dir='./models',
                     model_config_dir='./models/cloob_configs'):
    model_paths = {'cloob_RN50': join(model_dir, 'cloob_rn50_yfcc_epoch_28.pt'),
                   'cloob_RN50x4': join(model_dir, 'cloob_rn50x4_yfcc_epoch_28.pt')}
    checkpoint_path = model_paths[args.load_base_model]
    checkpoint = torch.load(checkpoint_path)
    model_config_file = os.path.join(model_config_dir,
                                     checkpoint['model_config_file'])
    
    device = args.device
    print("Device is ", device)

    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        _model_info = json.load(f)
        model_info = copy.deepcopy(_model_info)
        for k in _model_info:
            if 'cfg' in k:  # add vision_layers
                for _k, v in _model_info[k].items():
                    model_info[f'{k[:-4]}_{_k}'] = v
                    model_info[_k] = v
            model_info['image_resolution'] = model_info['vision_cfg']['image_size']
        print(model_info)

    model = CLIPGeneral(**model_info)
    preprocess = _transform(model.visual.input_resolution)

    if not torch.cuda.is_available():
        model.float()
    else:
        model.to(device)

    sd = checkpoint["state_dict"]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    if 'logit_scale_hopfield' in sd:
        sd.pop('logit_scale_hopfield', None)
    model.load_state_dict(sd)
    
    return model, preprocess
