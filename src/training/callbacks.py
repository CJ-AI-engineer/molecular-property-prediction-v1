"""
Callbacks for training loop.
Provides hooks for custom behavior during training.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch


class Callback:
    """Base class for callbacks."""
    
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch_idx):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx, logs):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    Stops training when monitored metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (whether lower or higher is better)
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Check if should stop training."""
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                print(f"\nEarly stopping triggered at epoch {epoch}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to only save best model
            save_freq: Frequency of saving (in epochs)
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Save checkpoint if conditions met."""
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        should_save = False
        
        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best_value
            else:
                improved = current > self.best_value
            
            if improved:
                self.best_value = current
                should_save = True
        else:
            should_save = (epoch % self.save_freq == 0)
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_value': self.best_value,
                'logs': logs,
            }
            

            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.filepath)
            print(f"Saved checkpoint to {self.filepath}")


class LearningRateLogger(Callback):
    """Log learning rate at each epoch."""
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Log current learning rate."""
        lr = trainer.optimizer.param_groups[0]['lr']
        logs['learning_rate'] = lr


class GradientLogger(Callback):
    """Log gradient statistics."""
    
    def __init__(self, log_freq: int = 10):
        """
        Initialize gradient logger.
        
        Args:
            log_freq: Frequency of logging (in batches)
        """
        self.log_freq = log_freq
    
    def on_batch_end(self, trainer, batch_idx, logs):
        """Log gradient statistics."""
        if batch_idx % self.log_freq == 0:
            # Compute gradient norm
            total_norm = 0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            logs['grad_norm'] = total_norm


class MetricLogger(Callback):
    """
    Log metrics to file.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize metric logger.
        
        Args:
            filepath: Path to log file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'w') as f:
            f.write("epoch,metric,value\n")
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Log metrics to file."""
        with open(self.filepath, 'a') as f:
            for metric, value in logs.items():
                f.write(f"{epoch},{metric},{value}\n")


class ProgressCallback(Callback):
    """Print training progress."""
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Print epoch summary."""
        train_loss = logs.get('train_loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        for key, value in logs.items():
            if key not in ['train_loss', 'val_loss', 'learning_rate']:
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    print("Callback classes defined.")
    print("\nAvailable callbacks:")
    print("  - EarlyStopping")
    print("  - ModelCheckpoint")
    print("  - LearningRateLogger")
    print("  - GradientLogger")
    print("  - MetricLogger")
    print("  - ProgressCallback")
