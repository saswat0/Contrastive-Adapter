"""
Miscellaneous utilities + setup
"""
import os
from os.path import join


def initialize_save_paths(args):
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    args.results_dir = join(f'./results/{args.dataset}')
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    args.results_dir = join(args.results_dir, args.config)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    args.results_dir = join(args.results_dir, 'embedding_adapter')
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if args.directory_name is not None:
        args.results_dir = join(args.results_dir, args.directory_name)
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        
    # Save models
    if not os.path.exists('./models/'):
        os.makedirs('./models/')
    args.model_dir = join(f'./models/{args.dataset}')
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    args.model_dir = join(args.model_dir, args.config)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    args.model_dir = join(args.model_dir, 'embedding_adapter')
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if args.directory_name is not None:
        args.model_dir = join(args.model_dir, args.directory_name)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
    args.model_path = join(args.model_dir,
                           f'{args.experiment_name}.pt')
    
    # Save embeddings
    if not os.path.exists('./embeddings/'):
        os.makedirs('./embeddings/')
    args.embeddings_dir = join(f'./embeddings/{args.dataset}')
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
    args.embeddings_dir = join(args.embeddings_dir, args.config)
    if not os.path.exists(args.embeddings_dir):
        os.makedirs(args.embeddings_dir)
        
        
def initialize_experiment(args):
    if args.dataset in ['waterbirds', 'celebA']:
        args.zeroshot_predict_by = 'kmeans_umap'
    else:
        args.zeroshot_predict_by = 'text'
    
    if args.zeroshot_predict_by == 'kmeans_umap':
        zeroshot_predict_by = 'kmu'
    else:
        zeroshot_predict_by = args.zeroshot_predict_by
        
    base_model = args.load_base_model.replace('/', '_').replace('-', '_')
        
    train_method_ = args.train_method if args.train_method != 'adapter_mt' else f'adapter_mt_nel={args.num_encoder_layers}'
    train_method = f'm={train_method_}-{base_model}_{zeroshot_predict_by}-hd={args.hidden_dim}-ahbn={int(args.adapter_head_batch_norm)}-rc={int(args.residual_connection)}-act={args.activation}'

    train_method += f'-ct={args.classification_temperature}-cm={args.contrastive_method}-bpc={int(args.balance_positive_classes)}-bac={int(args.balance_anchor_classes)}-np={args.num_positives}-nn={args.num_negatives}-nnm={args.nearest_neighbors_metric[:3]}-nnn={args.num_neighbors_for_negatives}-t={args.temperature}-gs={args.group_size}-tsr={args.train_sample_ratio}'

    args.train_method_verbose = train_method.replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')

    args.experiment_name = f'ca-{train_method}-me={args.max_epochs}-o={args.optimizer}-bs_trn={args.bs_trn}-lr={args.lr}_wd={args.weight_decay}_mo={args.momentum}-r={args.replicate}-s={args.seed}'
    args.config = f'config-'
    confounders = '_'.join(args.confounder_names)
    args.config += f'target={args.target_name}-confounders={confounders}'

    initialize_save_paths(args)

    args.results_path = join(args.results_dir,
                             f'r-{args.experiment_name}.csv')
    best_loss_model_path = join(args.model_dir, 
                                f'm-best_loss-tm={train_method}-a={args.arch}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}.pt')
    best_loss_model_path = best_loss_model_path.replace('EleutherAI_', '').replace('facebook_', '').replace('supcon', 'sc')
    print(f'-> Results dir:     {args.results_dir}')
    print(f'-> Model dir:       {args.model_dir}')
    print(f'-> Dataset:         {args.dataset} ({args.config})')
    print(f'-> Experiment:      {args.experiment_name}')
    print(f'-> Best model path: {best_loss_model_path}')
    print(f'---')
