"""
Checkpoint management utilities.
Save and load model checkpoints with full state.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


class CheckpointManager:
    """
    Manage model checkpoints.
    
    Handles saving/loading models with optimizer state, scheduler state,
    metrics, and any other training state.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
            save_best_only: Whether to only keep best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        self.best_metric = None
        self.checkpoints = []  # Thius is a List of (path, metric) tuples
    
    def save_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch: int = 0,
        metric: Optional[float] = None,
        is_best: bool = False,
        **kwargs
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            metric: Metric value (for tracking best)
            is_best: Whether this is the best model
            **kwargs: Additional items to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric': metric,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Add optimizer state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        # Fix up the filename
        if is_best or self.save_best_only:
            filename = 'best_model.pt'
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'checkpoint_epoch_{epoch}_{timestamp}.pt'
        
        filepath = self.checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        
        if not self.save_best_only:
            self.checkpoints.append((filepath, metric))
            self._cleanup_old_checkpoints()
        
        # Update best metric
        if is_best and metric is not None:
            self.best_metric = metric
        
        return filepath
    
    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        model,
        optimizer=None,
        scheduler=None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to
            
        Returns:
            Checkpoint dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load best checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to
            
        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        return self.load_checkpoint(best_path, model, optimizer, scheduler, device)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if self.max_checkpoints <= 0:
            return
        
        self.checkpoints.sort(key=lambda x: x[1] if x[1] is not None else float('-inf'))
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            path, _ = self.checkpoints.pop(0)
            if path.exists() and path.name != 'best_model.pt':
                path.unlink()
    
    def list_checkpoints(self) -> list:
        """
        List all checkpoint files.
        
        Returns:
            List of checkpoint paths
        """
        return sorted(self.checkpoint_dir.glob('*.pt'))
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        return checkpoints[-1]


def save_checkpoint(
    filepath: Union[str, Path],
    model,
    optimizer=None,
    scheduler=None,
    **kwargs
):
    """
    Quick checkpoint save function.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
        **kwargs: Additional items to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint.update(kwargs)
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model,
    optimizer=None,
    scheduler=None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Quick checkpoint load function.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        scheduler: Scheduler to load into (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


if __name__ == "__main__":
    import tempfile
    import torch.nn as nn
    
    print("Testing checkpoint utilities...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n1. Testing CheckpointManager")
        
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            max_checkpoints=3,
            save_best_only=False
        )
        
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(5):
            metric = 0.9 - epoch * 0.05
            is_best = (epoch == 2)
            
            path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric=metric,
                is_best=is_best,
                additional_info=f"Epoch {epoch}"
            )
            
            print(f"   Saved checkpoint for epoch {epoch}: {path.name}")
        
        print("\n   All checkpoints:")
        for ckpt in manager.list_checkpoints():
            print(f"     {ckpt.name}")
        
        print("\n2. Testing Load Checkpoint")
        
        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load best checkpoint
        checkpoint = manager.load_best_checkpoint(
            model=new_model,
            optimizer=new_optimizer
        )
        
        print(f"   Loaded checkpoint from epoch: {checkpoint['epoch']}")
        print(f"   Metric: {checkpoint['metric']:.4f}")
        print(f"   Timestamp: {checkpoint['timestamp']}")
        
        print("\n3. Testing Quick Save/Load Functions")
        
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        

        ckpt_path = Path(tmpdir) / 'quick_checkpoint.pt'
        save_checkpoint(
            filepath=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=10,
            loss=0.25
        )
        
        print(f"   Saved checkpoint: {ckpt_path}")
        
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        checkpoint = load_checkpoint(
            filepath=ckpt_path,
            model=new_model,
            optimizer=new_optimizer
        )
        
        print(f"   Loaded checkpoint:")
        print(f"     Epoch: {checkpoint['epoch']}")
        print(f"     Loss: {checkpoint['loss']}")
        
        print("\n4. Testing State Preservation")
        
        x = torch.randn(5, 10)
        y_before = model(x)
        
        save_checkpoint(ckpt_path, model)
        
        new_model = SimpleModel()
        load_checkpoint(ckpt_path, new_model)
        
        y_after = new_model(x)
        
        print(f"   Outputs match: {torch.allclose(y_before, y_after)}")
        
        print("\n5. Testing Get Latest Checkpoint")
        
        latest = manager.get_latest_checkpoint()
        if latest:
            print(f"   Latest checkpoint: {latest.name}")
        
        print("\n6. Testing Max Checkpoints Cleanup")
        print(f"   Max checkpoints: {manager.max_checkpoints}")
        print(f"   Number of checkpoints: {len(manager.list_checkpoints())}")
        
    print("\n Checkpoint tests complete!")
