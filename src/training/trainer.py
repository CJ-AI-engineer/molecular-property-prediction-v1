"""
Main training loop for molecular property prediction models.
Handles training, validation, checkpointing, and logging.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .losses import get_loss_fn
from .metrics import MetricCalculator


class MolecularPropertyTrainer:
    """
    Trainer for molecular property prediction models.
    
    Handles:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metric tracking
    - Optional experiment logging (MLflow, W&B)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        task_type: str = 'classification',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './checkpoints',
        exp_name: str = 'experiment',
        use_wandb: bool = False,
        use_mlflow: bool = False,
        gradient_clip: Optional[float] = None,
        patience: int = 20,
        min_delta: float = 0.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            optimizer: Optimizer (if None, uses Adam with lr=0.001)
            scheduler: Learning rate scheduler (optional)
            loss_fn: Loss function (if None, inferred from task_type)
            task_type: 'classification' or 'regression'
            device: Device to train on
            save_dir: Directory to save checkpoints
            exp_name: Experiment name
            use_wandb: Whether to log to Weights & Biases
            use_mlflow: Whether to log to MLflow
            gradient_clip: Gradient clipping value (None = no clipping)
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.task_type = task_type
        self.gradient_clip = gradient_clip
        self.patience = patience
        self.min_delta = min_delta
        
        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        
        # Loss function
        if loss_fn is None:
            self.loss_fn = get_loss_fn(task_type)
        else:
            self.loss_fn = loss_fn
        
        # Metrics
        self.metric_calculator = MetricCalculator(task_type=task_type)
        
        self.save_dir = Path(save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.save_dir / 'best_model.pt'
        self.last_model_path = self.save_dir / 'last_model.pt'
        
        # Tracking
        self.current_epoch = 0
        self.best_val_metric = float('inf') if task_type == 'regression' else 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self.exp_name = exp_name
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project="molecular-property-prediction", name=exp_name)
                wandb.watch(model)
            except ImportError:
                print("Warning: wandb not installed. Skipping W&B logging.")
                self.use_wandb = False
        
        if use_mlflow:
            try:
                import mlflow
                mlflow.start_run(run_name=exp_name)
                mlflow.log_params({
                    'model': type(model).__name__,
                    'task_type': task_type,
                    'optimizer': type(self.optimizer).__name__,
                })
            except ImportError:
                print("Warning: mlflow not installed. Skipping MLflow logging.")
                self.use_mlflow = False
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Make sure to predictions and targets have compatible shapes
            if predictions.dim() == 2 and predictions.size(1) == 1:
                predictions = predictions.squeeze(-1)
            if batch.y.dim() == 2 and batch.y.size(1) == 1:
                targets = batch.y.squeeze(-1)
            else:
                targets = batch.y
            
            # Compute loss
            loss = self.loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Store predictions and targets
            if self.task_type == 'classification':
                all_predictions.append(torch.sigmoid(predictions).detach().cpu())
            else:
                all_predictions.append(predictions.detach().cpu())
            all_targets.append(batch.y.detach().cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute average loss
        avg_loss = total_loss / len(self.train_loader)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metric_calculator.compute_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def validate(self, loader: Optional[DataLoader] = None) -> tuple:
        """
        Validate model.
        
        Args:
            loader: Data loader (if None, uses self.val_loader)
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Make sure to predictions and targets have compatible shapes
                if predictions.dim() == 2 and predictions.size(1) == 1:
                    predictions_for_loss = predictions.squeeze(-1)
                else:
                    predictions_for_loss = predictions
                if batch.y.dim() == 2 and batch.y.size(1) == 1:
                    targets_for_loss = batch.y.squeeze(-1)
                else:
                    targets_for_loss = batch.y
                
                # Compute loss
                loss = self.loss_fn(predictions_for_loss, targets_for_loss)
                total_loss += loss.item()
                
                # Store predictions and targets
                if self.task_type == 'classification':
                    all_predictions.append(torch.sigmoid(predictions).cpu())
                else:
                    all_predictions.append(predictions.cpu())
                all_targets.append(batch.y.cpu())
        
        # Compute average loss
        avg_loss = total_loss / len(loader)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metric_calculator.compute_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def _get_main_metric(self, metrics: Dict[str, float]) -> float:
        """Get the main metric for model selection."""
        if self.task_type == 'classification':
            return metrics.get('roc_auc', metrics.get('accuracy', 0.0))
        else:
            return metrics.get('rmse', metrics.get('mae', float('inf')))
    
    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is an improvement."""
        if self.task_type == 'classification':
            return current_metric > (self.best_val_metric + self.min_delta)
        else:
            return current_metric < (self.best_val_metric - self.min_delta)
    
    def _save_checkpoint(self, path: Path, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
    
    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float,
                    train_metrics: Dict, val_metrics: Dict):
        """Log metrics to experiment trackers."""
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        for name, value in val_metrics.items():
            print(f"  Val {name}: {value:.4f}")
        
        if self.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            })
        
        # MLflow logging
        if self.use_mlflow:
            import mlflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
            }, step=epoch)
    
    def train(self, num_epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Log metrics
            if verbose:
                self._log_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Check for improvement
            main_metric = self._get_main_metric(val_metrics)
            
            if self._is_improvement(main_metric):
                self.best_val_metric = main_metric
                self.patience_counter = 0
                
                # Save best model
                self._save_checkpoint(self.best_model_path, is_best=True)
                
                if verbose:
                    print(f"   New best model! Metric: {main_metric:.4f}")
            else:
                self.patience_counter += 1
                
                if verbose:
                    print(f"  No improvement ({self.patience_counter}/{self.patience})")
            
            # Save last model
            self._save_checkpoint(self.last_model_path, is_best=False)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(main_metric)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Best validation metric: {self.best_val_metric:.4f}")
            print(f"Best model saved to: {self.best_model_path}")
        
        # Final evaluation on test set
        if self.test_loader is not None:
            test_loss, test_metrics = self.evaluate(self.test_loader)
            self.history['test_metrics'] = test_metrics
            
            if verbose:
                print(f"\nTest Results:")
                for name, value in test_metrics.items():
                    print(f"  {name}: {value:.4f}")
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        if self.use_mlflow:
            import mlflow
            mlflow.end_run()
        
        return self.history
    
    def evaluate(self, loader: DataLoader) -> tuple:
        """
        Evaluate model on a dataset.
        
        Args:
            loader: Data loader
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        return self.validate(loader)
    
    def predict(self, loader: DataLoader) -> torch.Tensor:
        """
        Make predictions on a dataset.
        
        Args:
            loader: Data loader
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                batch = batch.to(self.device)
                predictions = self.model(batch)
                
                if self.task_type == 'classification':
                    predictions = torch.sigmoid(predictions)
                
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def load_best_model(self):
        """Load the best model from checkpoint."""
        if not self.best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {self.best_model_path}")
        
        checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
        print(f"Best validation metric: {checkpoint['best_val_metric']:.4f}")


if __name__ == "__main__":
    print("Trainer class defined. See scripts/train.py for usage example.")

