#!/usr/bin/env python3
"""
Evaluate trained models on test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset BBBP
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data-root data/processed
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from sklearn.metrics import classification_report

from src.data import MolecularDataset, get_dataloader
from src.models import get_model
from src.training import MetricCalculator
from src.utils import (
    load_checkpoint,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_distribution
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate molecular property prediction model')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='BBBP', help='Dataset name')
    parser.add_argument('--data-root', type=str, default='data/processed', help='Data root directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--task-type', type=str, default='classification', 
                       choices=['classification', 'regression'], help='Task type')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (None = CPU)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Classification threshold')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.gpu is not None and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = MolecularDataset(root=args.data_root, dataset_name=args.dataset)
    
    test_loader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test samples: {len(dataset)}")
    
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    from src.models import MolecularGCN
    model = MolecularGCN(
        node_feat_dim=dataset.num_node_features,
        edge_feat_dim=dataset.num_edge_features,
        hidden_dim=128, 
        num_tasks=dataset.num_tasks
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("\nRunning inference...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions = model(batch)
            
            if args.task_type == 'classification':
                predictions = torch.sigmoid(predictions)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch.y.cpu())
    
    predictions = torch.cat(all_predictions, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    metric_calc = MetricCalculator(task_type=args.task_type)
    metrics = metric_calc.compute_metrics(targets, predictions)
    
    print("\nTest Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    if args.task_type == 'classification':
        plot_confusion_matrix(
            targets.flatten(),
            (predictions >= args.threshold).astype(int).flatten(),
            save_path=output_dir / 'confusion_matrix.png',
            show=False
        )
        plot_roc_curve(
            targets.flatten(),
            predictions.flatten(),
            save_path=output_dir / 'roc_curve.png',
            show=False
        )
    
    plot_prediction_distribution(
        targets.flatten(),
        predictions.flatten(),
        task_type=args.task_type,
        save_path=output_dir / 'predictions.png',
        show=False
    )
    
    np.savez(output_dir / 'predictions.npz', predictions=predictions, targets=targets)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
