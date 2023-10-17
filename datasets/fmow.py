"""
Functional Map of the World (FMoW) Dataset

Setup from https://github.com/p-lambda/wilds/blob/main/wilds/datasets/fmow_dataset.py

Relies on the WILDS module, which can be installed with `pip install wilds`
Alternatively, pull the main branch from GitHub and include the `wilds/wilds` directory in this project directory.  
"""

import os
from os.path import join
from copy import deepcopy

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class FMoW(Dataset):
    """
    FMoW dataset
    """
    
    def __init__(self, root_dir, target_name, confounder_names,
                 args=None, transform=None):
        
        self.root_dir = join(root_dir, 'fmow_v1.1')
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.args = args
        self.transform = transform
        
        self.df_country_code_mapping = pd.read_csv(join(self.root_dir, 'country_code_mapping.csv'))
        self.df_metadata = pd.read_csv(join(self.root_dir, 'rgb_metadata.csv'))
        
        self.countrycode_to_region = {k: v for k, v in zip(self.df_country_code_mapping['alpha-3'], self.df_country_code_mapping['region'])}
        self.regions = [self.countrycode_to_region.get(code, 'Other') for code in self.df_metadata['country_code'].to_list()]
        self.df_metadata['region'] = self.regions
        self.all_countries = self.df_metadata['country_code']
        
        # Main dataset courtesy of WILDS
        self.wilds_dataset = get_dataset(dataset='fmow', download=False,
                                         root_dir=root_dir)
        
        self.categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]
        self.categories = [c.replace('_', ' ') 
                           for c in self.categories]
        
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        self.prompt_templates = ['satellite imagery of {}.', 
                                 'aerial imagery of {}.', 
                                 'satellite photo of {}.',
                                 'an aerial photo of {}.',
                                 'satellite view of {}.',
                                 'aerial view of {}.',
                                 'satellite imagery of a {}.',
                                 'aerial imagery of a {}.',
                                 'a satellite photo of a {}.',
                                 'an aerial photo of a {}.',
                                 'a satellite view of a {}.',
                                 'an aerial view of a {}.',
                                 'satellite imagery of the {}.',
                                 'aerial imagery of the {}.',
                                 'a satellite photo of the {}.',
                                 'an aerial photo of the {}.',
                                 'a satellite view of the {}.',
                                 'an aerial view of the {}.']

    
def initialize_targets(dataloader):
    region = dataloader.dataset.metadata_array[:, 0].numpy().astype(str)
    year   = dataloader.dataset.metadata_array[:, 1].numpy().astype(str)
    spurious = year + np.array(['_'] * len(year)).astype('object') + region
    spurious_to_idx = {cat: i for i, cat in enumerate(np.unique(spurious))}
    targets_t = dataloader.dataset.y_array.numpy()
    targets_s = np.asarray([spurious_to_idx[y] for y in list(spurious)])

    targets = dataloader.dataset.y_array
    targets_all = {'target': targets_t, 'spurious': targets_s}
    return targets, targets_all
        

def load_fmow(args, train_shuffle=True, transform=None):
    dataset = FMoW(args.root_dir,
                   target_name='building_land',
                   confounder_names=['year', 'region'],
                   args=args, transform=transform)
    dataloaders = []
    for ix, split in enumerate(['train', 'id_val', 'val', 'id_test', 'test']):
        batch_size = args.bs_trn if ix == 0 else args.bs_val
        shuffle = True if (train_shuffle and 'split' == 'train') else False
        _dataset = dataset.wilds_dataset.get_subset(split, transform=transform)
        dataloader = DataLoader(_dataset, shuffle=shuffle, sampler=None,
                                collate_fn=_dataset.collate, batch_size=batch_size)
        targets, targets_all = initialize_targets(dataloader)
        dataloader.dataset.targets = targets
        dataloader.dataset.targets_all = targets_all
        # CLIP Prompts
        dataloader.dataset.classes = dataset.categories
        dataloader.dataset.prompts = dataset.prompt_templates
        # WILDS Eval
        dataloader.dataset.eval = dataset.wilds_dataset.eval
        dataloader.wilds_dataset = dataset.wilds_dataset
        dataloaders.append(dataloader)
    return dataloaders


def visualize_fmow(dataloader, num_datapoints, title, args, save,
                   save_id, ftype='png', target_type='target'):
    # Filter for selected datapoints (in case we use SubsetRandomSampler)
    try:
        subset_indices = dataloader.sampler.indices
        targets = dataloader.dataset.targets_all[target_type][subset_indices]
        subset = True
    except AttributeError:
        targets = dataloader.dataset.targets_all[target_type]
        subset = False
    all_data_indices = []
    for class_ in np.unique(targets):
        class_indices = np.where(targets == class_)[0]
        if subset:
            class_indices = subset_indices[class_indices]
        all_data_indices.extend(class_indices[:num_datapoints])
    
    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in
                     all_data_indices],
                    mean=np.mean([0.485, 0.456, 0.406]),  
                    std=np.mean([0.229, 0.224, 0.225]), nrow=8, title=title,
                    args=args, save=save, save_id=save_id, ftype=ftype)



def load_dataloaders(args, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_fmow(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                      save_id, ftype='png', target_type='target'):
    return visualize_fmow(dataloader, num_datapoints, title,
                          args, save, save_id, ftype, target_type)
