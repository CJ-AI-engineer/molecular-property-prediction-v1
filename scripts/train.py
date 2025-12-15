#!/usr/bin/env python3


import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.data import (
    MolecularDataset,
    get_dataloader,
    get_splitter
)
from src.models import get_model
from src.training import (
    MolecularPropertyTrainer,
    get_optimizer,
    get_scheduler,
    get_loss_fn
)
from src.utils import (
    load_config,
    get_default_config,
    set_seed,
    ExperimentLogger,
    plot_training_curves,
    plot_metrics,
    plot_roc_curve,
    plot_prediction_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train molecular property prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (YAML/JSON)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='BBBP',
        help='Dataset name (BBBP, HIV, ESOL, FreeSolv, QM9)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed',
        help='Root directory for processed data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--split-type',
        type=str,
        default='random',
        choices=['random', 'scaffold', 'stratified', 'temporal'],
        help='Data splitting strategy'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gcn',
        choices=['gcn', 'gin', 'gat', 'gin_edge'],
        help='Model architecture'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=5,
        help='Number of graph layers'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout probability'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='mean',
        choices=['mean', 'add', 'max', 'attention', 'set2set'],
        help='Graph pooling method'
    )
    
    parser.add_argument(
        '--task-type',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='Task type'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
        help='Optimizer'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='reduce_on_plateau',
        choices=['none', 'step', 'exponential', 'reduce_on_plateau', 'cosine', 'one_cycle'],
        help='Learning rate scheduler'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--gradient-clip',
        type=float,
        default=1.0,
        help='Gradient clipping value (0 = no clipping)'
    )
    
    # Experiment
    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory to save logs'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory to save results and plots'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Experiment tracking
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--use-mlflow',
        action='store_true',
        help='Use MLflow for logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='molecular-property-prediction',
        help='W&B project name'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID to use (None = CPU, -1 = all available GPUs)'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate model (requires --resume)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def setup_device(args):
    """Setup device for training."""
    if args.no_cuda:
        return 'cpu'
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return 'cpu'
    
    if args.gpu is None:
        # Use first GPU which is available
        device = 'cuda:0'
    elif args.gpu == -1:
        # Use all GPUs (DataParallel)
        device = 'cuda'
    else:
        device = f'cuda:{args.gpu}'
    
    print(f"Using device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return device


def load_data(args, logger):
    """Load and split dataset."""
    logger.info("=" * 60)
    logger.info("Loading Dataset")
    logger.info("=" * 60)
    
    # Load dataset
    try:
        dataset = MolecularDataset(
            root=args.data_root,
            dataset_name=args.dataset
        )
    except FileNotFoundError:
        logger.error(f"Dataset {args.dataset} not found!")
        logger.error(f"Please run: python scripts/preprocess_data.py --dataset {args.dataset}")
        sys.exit(1)
    
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Node features: {dataset.num_node_features}")
    logger.info(f"Edge features: {dataset.num_edge_features}")
    logger.info(f"Tasks: {dataset.num_tasks}")
    
    # Split dataset
    logger.info(f"\nSplitting data using {args.split_type} strategy...")
    splitter = get_splitter(args.split_type)
    
    train_idx, val_idx, test_idx = splitter.split(
        dataset,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=args.seed
    )
    
    logger.info(f"Train: {len(train_idx)} samples")
    logger.info(f"Val: {len(val_idx)} samples")
    logger.info(f"Test: {len(test_idx)} samples")
    
    # Create data loaders
    train_loader = get_dataloader(
        dataset[train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = get_dataloader(
        dataset[val_idx],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loader = get_dataloader(
        dataset[test_idx],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return dataset, train_loader, val_loader, test_loader


def build_model(args, dataset, logger, device):
    """Build model."""
    logger.info("=" * 60)
    logger.info("Building Model")
    logger.info("=" * 60)
    
    model = get_model(
        model_type=args.model,
        node_feat_dim=dataset.num_node_features,
        edge_feat_dim=dataset.num_edge_features,
        hidden_dim=args.hidden_dim,
        num_tasks=dataset.num_tasks,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling
    )
    
    model = model.to(device)
    

    logger.log_model_info(model)
    
    return model


def create_trainer(args, model, train_loader, val_loader, test_loader, logger, device):
    """Create trainer with all components."""
    logger.info("=" * 60)
    logger.info("Setting up Training")
    logger.info("=" * 60)
    

    optimizer = get_optimizer(
        model.parameters(),
        optimizer_type=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    
    # Scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        num_epochs=args.epochs
    )
    if scheduler is not None:
        logger.info(f"Scheduler: {args.scheduler}")
    
    # Loss function
    loss_fn = get_loss_fn(args.task_type)
    logger.info(f"Loss function: {type(loss_fn).__name__}")
    
    trainer = MolecularPropertyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        task_type=args.task_type,
        device=device,
        save_dir=args.save_dir,
        exp_name=args.exp_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
        patience=args.patience
    )
    
    logger.info(f"Early stopping patience: {args.patience}")
    if args.gradient_clip > 0:
        logger.info(f"Gradient clipping: {args.gradient_clip}")
    
    return trainer


def train(args, trainer, logger):
    """Run training."""
    logger.info("=" * 60)
    logger.info("Training")
    logger.info("=" * 60)
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        verbose=args.verbose
    )
    
    return history


def evaluate(args, trainer, logger, results_dir):
    """Evaluate model and save results."""
    logger.info("=" * 60)
    logger.info("Evaluation")
    logger.info("=" * 60)
    
    # Load best model
    trainer.load_best_model()
    
    # Get predictions
    predictions = trainer.predict(trainer.test_loader)
    
    # Get true values
    all_targets = []
    for batch in trainer.test_loader:
        all_targets.append(batch.y.cpu())
    targets = torch.cat(all_targets, dim=0)
    
    # Evaluate
    test_loss, test_metrics = trainer.evaluate(trainer.test_loader)
    
    logger.log_results(test_metrics)
    
    # Save predictions
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pred_file = results_dir / f'{args.exp_name}_predictions.npz'
    np.savez(
        pred_file,
        predictions=predictions.numpy(),
        targets=targets.numpy()
    )
    logger.info(f"Saved predictions to {pred_file}")
    
    return test_metrics, predictions, targets


def visualize_results(args, history, predictions, targets, logger, results_dir):
    """Create and save visualizations."""
    logger.info("=" * 60)
    logger.info("Creating Visualizations")
    logger.info("=" * 60)
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Training curves
    logger.info("Plotting training curves...")
    plot_training_curves(
        history,
        save_path=results_dir / f'{args.exp_name}_training_curves.png',
        show=False
    )
    
    # Metrics
    if args.task_type == 'classification':
        metrics_to_plot = ['roc_auc', 'accuracy', 'f1']
    else:
        metrics_to_plot = ['rmse', 'mae', 'r2']
    
    logger.info(f"Plotting metrics: {metrics_to_plot}...")
    plot_metrics(
        history,
        metrics=metrics_to_plot,
        save_path=results_dir / f'{args.exp_name}_metrics.png',
        show=False
    )
    
 
    if args.task_type == 'classification':
        logger.info("Plotting ROC curve...")
        plot_roc_curve(
            targets.numpy(),
            predictions.numpy(),
            save_path=results_dir / f'{args.exp_name}_roc_curve.png',
            show=False
        )
    

    logger.info("Plotting prediction distribution...")
    plot_prediction_distribution(
        targets.numpy(),
        predictions.numpy(),
        task_type=args.task_type,
        save_path=results_dir / f'{args.exp_name}_predictions.png',
        show=False
    )
    
    logger.info(f"Saved all plots to {results_dir}/")


def main():
    """Main training function."""
    args = parse_args()
    
    if args.config is not None:
        config = load_config(args.config)
        
        # Flatten nested config structure
        flat_config = {}
        
        if 'model' in config._config:
            for k, v in config._config['model'].items():
                flat_config[k.replace('_', '-')] = v
            flat_config['model'] = config._config['model'].get('type', 'gcn')
        
        # Data config
        if 'data' in config._config:
            flat_config['dataset'] = config._config['data'].get('dataset', 'BBBP')
            flat_config['data_root'] = config._config['data'].get('data_root', 'data/processed')
            flat_config['batch_size'] = config._config['data'].get('batch_size', 32)
            flat_config['num_workers'] = config._config['data'].get('num_workers', 4)
            flat_config['split_type'] = config._config['data'].get('split_type', 'random')
        
        # Training config
        if 'training' in config._config:
            flat_config['task_type'] = config._config['training'].get('task_type', 'classification')
            flat_config['epochs'] = config._config['training'].get('epochs', 100)
            flat_config['patience'] = config._config['training'].get('patience', 20)
            flat_config['gradient_clip'] = config._config['training'].get('gradient_clip', 1.0)
        
        # Optimizer config
        if 'optimizer' in config._config:
            flat_config['optimizer'] = config._config['optimizer'].get('type', 'adam')
            flat_config['lr'] = config._config['optimizer'].get('lr', 0.001)
            flat_config['weight_decay'] = config._config['optimizer'].get('weight_decay', 1e-5)
        
        # Scheduler config
        if 'scheduler' in config._config:
            flat_config['scheduler'] = config._config['scheduler'].get('type', 'reduce_on_plateau')
        
        # Experiment config - ask Michelle Madam, if it will be used later
        if 'experiment' in config._config:
            flat_config['exp_name'] = config._config['experiment'].get('name', None)
            flat_config['save_dir'] = config._config['experiment'].get('save_dir', './checkpoints')
            flat_config['log_dir'] = config._config['experiment'].get('log_dir', './logs')
            flat_config['results_dir'] = config._config['experiment'].get('results_dir', './results')
            flat_config['seed'] = config._config['experiment'].get('seed', 42)
            flat_config['use_wandb'] = config._config['experiment'].get('use_wandb', False)
            flat_config['use_mlflow'] = config._config['experiment'].get('use_mlflow', False)
            flat_config['wandb_project'] = config._config['experiment'].get('wandb_project', 'molecular-property-prediction')
        
        # Device config
        if 'device' in config._config:
            flat_config['gpu'] = config._config['device'].get('gpu', None)
            flat_config['no_cuda'] = config._config['device'].get('no_cuda', False)
        

        for key, value in vars(args).items():
            if value is not None and key != 'config':
                flat_config[key] = value
        
        # Update args with flattened config
        for key, value in flat_config.items():
            setattr(args, key, value)
    
    # Generate experiment name if not provided - this causes problem, ask Madam if this feature is really needed
    if not hasattr(args, 'exp_name') or args.exp_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'{args.dataset}_{args.model}_{timestamp}'
    
    set_seed(args.seed)
    
    device = setup_device(args)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = ExperimentLogger(
        exp_name=args.exp_name,
        log_dir=args.log_dir
    )
    
    logger.info("=" * 60)
    logger.info("Experiment Configuration")
    logger.info("=" * 60)
    logger.log_params(vars(args))
    
    dataset, train_loader, val_loader, test_loader = load_data(args, logger)
    
    model = build_model(args, dataset, logger, device)
    
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        from src.utils import load_checkpoint
        checkpoint = load_checkpoint(args.resume, model, device=device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    

    if args.evaluate_only:
        if args.resume is None:
            logger.error("--resume must be specified for evaluation mode")
            sys.exit(1)
        
        trainer = create_trainer(
            args, model, train_loader, val_loader, test_loader, logger, device
        )
        
        results_dir = Path(args.results_dir) / args.exp_name
        test_metrics, predictions, targets = evaluate(args, trainer, logger, results_dir)
        visualize_results(args, {}, predictions, targets, logger, results_dir)
        
        logger.info("=" * 60)
        logger.info("Evaluation Complete!")
        logger.info("=" * 60)
        return
    
    trainer = create_trainer(
        args, model, train_loader, val_loader, test_loader, logger, device
    )
    
    history = train(args, trainer, logger)
    

    results_dir = Path(args.results_dir) / args.exp_name
    test_metrics, predictions, targets = evaluate(args, trainer, logger, results_dir)
    

    visualize_results(args, history, predictions, targets, logger, results_dir)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best model saved to: {trainer.best_model_path}")
    logger.info(f"Results saved to: {results_dir}/")
    logger.info(f"Logs saved to: {log_dir}/")
    
    logger.info("\nFinal Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()



"""
==== NOTE: 
Complete training script for molecular property prediction.

Usage:
    python scripts/train.py --config configs/gcn_config.yaml
    python scripts/train.py --dataset BBBP --model gcn --epochs 100
    python scripts/train.py --config configs/gin_config.yaml --gpu 0
"""

