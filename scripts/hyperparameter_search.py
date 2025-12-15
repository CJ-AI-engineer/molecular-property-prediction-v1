#!/usr/bin/env python3
"""
Hyperparameter search for molecular property prediction using Optuna.

Usage:
    python scripts/hyperparameter_search.py --dataset BBBP --model gcn --n-trials 50
"""

import os
import sys
import argparse
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False
    sys.exit(1)

from src.data import MolecularDataset, get_dataloader, get_splitter
from src.models import get_model
from src.training import (
    MolecularPropertyTrainer,
    get_optimizer,
    get_scheduler
)
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter search')
    
    parser.add_argument('--dataset', type=str, default='BBBP')
    parser.add_argument('--data-root', type=str, default='data/processed')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin', 'gat'])
    parser.add_argument('--task-type', type=str, default='classification')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=None)
    
    return parser.parse_args()


def objective(trial, args, dataset, train_loader, val_loader, device):
    """Objective function for Optuna."""
    
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 3, 7)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    

    train_loader = get_dataloader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_loader.dataset, batch_size=batch_size, shuffle=False)
    
    model = get_model(
        model_type=args.model,
        node_feat_dim=dataset.num_node_features,
        edge_feat_dim=dataset.num_edge_features,
        hidden_dim=hidden_dim,
        num_tasks=dataset.num_tasks,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), 'adam', lr, weight_decay)
    scheduler = get_scheduler(optimizer, 'reduce_on_plateau', patience=5)
    

    trainer = MolecularPropertyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        task_type=args.task_type,
        device=device,
        save_dir=f'./optuna_trials/trial_{trial.number}',
        exp_name=f'trial_{trial.number}',
        patience=10,
        use_wandb=False,
        use_mlflow=False
    )
    

    for epoch in range(args.n_epochs):
        trainer.train_epoch()
        _, val_metrics = trainer.validate()
        
        intermediate_value = val_metrics.get('roc_auc', 0.0) if args.task_type == 'classification' else -val_metrics.get('rmse', float('inf'))
        
        trial.report(intermediate_value, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    

    _, val_metrics = trainer.validate()
    return val_metrics.get('roc_auc', 0.0) if args.task_type == 'classification' else -val_metrics.get('rmse', float('inf'))


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = f'cuda:{args.gpu}' if args.gpu is not None and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    dataset = MolecularDataset(root=args.data_root, dataset_name=args.dataset)
    splitter = get_splitter('random')
    train_idx, val_idx, _ = splitter.split(dataset, seed=args.seed)
    
    train_loader = get_dataloader(dataset[train_idx], batch_size=32, shuffle=True)
    val_loader = get_dataloader(dataset[val_idx], batch_size=32, shuffle=False)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print(f"\nStarting hyperparameter search...")
    print(f"Trials: {args.n_trials}, Epochs per trial: {args.n_epochs}\n")
    
    study.optimize(
        lambda trial: objective(trial, args, dataset, train_loader, val_loader, device),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    

    print("\n" + "=" * 60)
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest value: {study.best_value:.4f}")
    


    results_dir = Path('./hyperparameter_results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")


if __name__ == '__main__':
    main()
