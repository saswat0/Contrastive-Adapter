# Code for: Contrastive Adapters for Foundation Model Group Robustness
This folder contains code for our NeurIPS 2022 submission.

## Requirements

Weinclude a `requirements.txt` file for installing dependencies with `pip install -r requirements.txt`.

List of (installable) dependencies:  
* python 3.7.9  
* matplotlib 3.3.2
* numpy 1.19.2  
* pandas 1.1.3  
* pillow 8.0.1  
* pytorch=1.7.0  
* scikit-learn 0.23.2  
* scipy 1.5.2  
* transformers 4.4.2 
* torchvision 0.8.1  
* tqdm 4.54.0  
* umap-learn 0.4.6

## Datasets and code 

**Waterbirds**: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). Unzipping this should result in a folder `waterbird_complete95_forest2water2`, which should be moved to `./datasets/data/Waterbirds/`.  

**CelebA**: Download dataset files from this [Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). Then move files to `./datasets/data/CelebA/` such that we have the following structure:
```
# In `./datasets/data/CelebA/`:
|-- list_attr_celeba.csv
|-- list_eval_partition.csv
|-- img_align_celeba/
    |-- image1.png
    |-- ...
    |-- imageN.png
```  

**BREEDS Datasets** (Living-17, Nonliving-26): Download files for a version of ImageNet (e.g. ILSVRC2012) and save to `./datasets/data/imagenet`.  

**CIFAR-10 Datasets** (CIFAR-10.001, CIFAR-10.02): Download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and save to `./datasets/data/cifar10/`. Download the CIFAR-10.1 dataset files from the official GitHub [repository](https://github.com/modestyachts/CIFAR-10.1) and save to `./datasets/data/cifar10.1/`. Download the CIFAR-10.2 dataset files from the official GitHub [repository](https://github.com/modestyachts/cifar-10.2) and save to `./datasets/data/cifar10.2/`.

**WILDS Datasets** (FMoW-WILDS, Amazon-WILDS, CivilComments-WILDS): We recommend following the setup instructions provided by the official WILDS [website](https://wilds.stanford.edu/get_started/). Individual dataset downloads and assets should then be placed in `./datasets/data/WILDS_datasets/`.


## Sample commands for training  

We provide sample scripts below.

### Waterbirds  
 
```
python main.py --dataset waterbirds --torch_seed --load_base_model clip_RN50 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layer 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100  --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 1
```

### CelebA

```
python main.py --dataset celebA --torch_seed --load_base_model clip_RN50 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layer 1 --hidden_dim 128 --max_epochs 50 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100  --temperature 0.1 --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 2
```

### BREEDS Living-17

```
python main.py --dataset breeds_living17_source_target --torch_seed --load_base_model clip_RN50 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layer 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100  --temperature 0.1 --balance_anchor_classes --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 2
```

### CIFAR-10.02
```
python main.py --dataset cifar10e2 --torch_seed --load_base_model clip_RN50 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layer 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 512 --num_negatives 512 --num_neighbors_for_negatives 1024 --classification_temperature 100  --temperature 0.1 --balance_anchor_classes --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 2
```



To change the foundation model architecture, specify `--load_base_model [arch]`. We support the following choices:  
```
# Vision  
--load_base_model clip_RN50  
--load_base_model clip_RN101
--load_base_model clip_RN50x4
--load_base_model clip_ViTB32
--load_base_model clip_ViTB16
--load_base_model clip_ViTL14 
--load_base_model clip_ViTL14b  # ViT-L/14@336px

--load_base_model cloob_RN50  
--load_base_model cloob_RN50x4

# Language
--load_base_model 'EleutherAI/gpt-neo-125M'
--load_base_model 'EleutherAI/gpt-neo-1.3B'
```
