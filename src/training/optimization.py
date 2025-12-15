"""
Optimizer and learning rate scheduler configurations.
"""

import torch
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    _LRScheduler, StepLR, ExponentialLR, ReduceLROnPlateau,
    CosineAnnealingLR, OneCycleLR, CyclicLR
)
from typing import Dict, Any, Optional


def get_optimizer(
    model_parameters,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Get optimizer by name.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD,
        'rmsprop': RMSprop,
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {optimizer_type}. "
            f"Choose from {list(optimizers.keys())}"
        )
    
    optimizer_class = optimizers[optimizer_type]
    
    params = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
    }
    
    # Optimizer-specific parameters
    if optimizer_type in ['adam', 'adamw']:
        params['betas'] = kwargs.get('betas', (0.9, 0.999))
        params['eps'] = kwargs.get('eps', 1e-8)
    
    elif optimizer_type == 'sgd':
        params['momentum'] = kwargs.get('momentum', 0.9)
        params['nesterov'] = kwargs.get('nesterov', True)
    
    elif optimizer_type == 'rmsprop':
        params['alpha'] = kwargs.get('alpha', 0.99)
        params['eps'] = kwargs.get('eps', 1e-8)
    
    return optimizer_class(model_parameters, **params)


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'reduce_on_plateau',
    num_epochs: Optional[int] = None,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Get learning rate scheduler by name.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs (for some schedulers)
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == 'none' or scheduler_type is None:
        return None
    
    schedulers = {
        'step': StepLR,
        'exponential': ExponentialLR,
        'reduce_on_plateau': ReduceLROnPlateau,
        'cosine': CosineAnnealingLR,
        'one_cycle': OneCycleLR,
        'cyclic': CyclicLR,
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {scheduler_type}. "
            f"Choose from {list(schedulers.keys())}"
        )
    
    scheduler_class = schedulers[scheduler_type]
    
    # Scheduler-specific parameters
    if scheduler_type == 'step':
        params = {
            'step_size': kwargs.get('step_size', 30),
            'gamma': kwargs.get('gamma', 0.1),
        }
    
    elif scheduler_type == 'exponential':
        params = {
            'gamma': kwargs.get('gamma', 0.95),
        }
    
    elif scheduler_type == 'reduce_on_plateau':
        params = {
            'mode': kwargs.get('mode', 'min'),
            'factor': kwargs.get('factor', 0.5),
            'patience': kwargs.get('patience', 10),
            'min_lr': kwargs.get('min_lr', 1e-6),
        }
    
    elif scheduler_type == 'cosine':
        if num_epochs is None:
            raise ValueError("num_epochs required for cosine annealing")
        params = {
            'T_max': num_epochs,
            'eta_min': kwargs.get('eta_min', 0),
        }
    
    elif scheduler_type == 'one_cycle':
        if num_epochs is None:
            raise ValueError("num_epochs required for one cycle")
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        params = {
            'max_lr': optimizer.param_groups[0]['lr'],
            'epochs': num_epochs,
            'steps_per_epoch': steps_per_epoch,
        }
    
    elif scheduler_type == 'cyclic':
        params = {
            'base_lr': optimizer.param_groups[0]['lr'] / 10,
            'max_lr': optimizer.param_groups[0]['lr'],
            'step_size_up': kwargs.get('step_size_up', 2000),
            'mode': kwargs.get('mode', 'triangular'),
        }
    
    return scheduler_class(optimizer, **params)


def configure_optimization(
    model_parameters,
    config: Dict[str, Any]
) -> tuple:
    """
    Configure optimizer and scheduler from config dict.
    
    Args:
        model_parameters: Model parameters
        config: Configuration dictionary with keys:
            - optimizer: optimizer config
            - scheduler: scheduler config  
            - training: training config
            
    Returns:
        Tuple of (optimizer, scheduler)
    """
    opt_config = config.get('optimizer', {})
    optimizer_type = opt_config.get('type', 'adam')
    learning_rate = opt_config.get('lr', 0.001)
    weight_decay = opt_config.get('weight_decay', 0.0)
    
    optimizer = get_optimizer(
        model_parameters,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        **opt_config
    )
    
    sched_config = config.get('scheduler', {})
    scheduler_type = sched_config.get('type', 'reduce_on_plateau')
    num_epochs = config.get('training', {}).get('epochs', None)
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        num_epochs=num_epochs,
        **sched_config
    )
    
    return optimizer, scheduler


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup.
    Gradually increases LR from 0 to initial LR over warmup_epochs.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        after_scheduler: Optional[_LRScheduler] = None
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            after_scheduler: Scheduler to use after warmup
        """
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.finished_warmup = True
                if self.after_scheduler is not None:
                    self.after_scheduler.base_lrs = self.base_lrs
            
            if self.after_scheduler is not None:
                return self.after_scheduler.get_last_lr()
            else:
                return self.base_lrs
    
    def step(self, epoch=None):
        if self.finished_warmup and self.after_scheduler is not None:
            self.after_scheduler.step(epoch)
        else:
            super().step(epoch)



if __name__ == "__main__":
    import torch.nn as nn
    
    print("Testing optimization utilities...")
    
    model = nn.Linear(10, 1)
    
    # Test 1: Get optimizers
    print("\n1. Testing Optimizers")
    for opt_type in ['adam', 'adamw', 'sgd', 'rmsprop']:
        optimizer = get_optimizer(
            model.parameters(),
            optimizer_type=opt_type,
            learning_rate=0.001
        )
        print(f"   {opt_type}: {type(optimizer).__name__}")
    
    # Test 2: Get schedulers
    print("\n2. Testing Schedulers")
    optimizer = get_optimizer(model.parameters())
    
    for sched_type in ['step', 'exponential', 'reduce_on_plateau', 'cosine']:
        scheduler = get_scheduler(
            optimizer,
            scheduler_type=sched_type,
            num_epochs=100
        )
        if scheduler is not None:
            print(f"   {sched_type}: {type(scheduler).__name__}")
    
    # Test 3: Configuration-based setup
    print("\n3. Testing Config-Based Setup")
    config = {
        'optimizer': {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5
        },
        'scheduler': {
            'type': 'reduce_on_plateau',
            'factor': 0.5,
            'patience': 10
        },
        'training': {
            'epochs': 100
        }
    }
    
    optimizer, scheduler = configure_optimization(model.parameters(), config)
    print(f"   Optimizer: {type(optimizer).__name__}")
    print(f"   Scheduler: {type(scheduler).__name__}")
    print(f"   Initial LR: {optimizer.param_groups[0]['lr']}")
    
    # Test 4: Warmup scheduler
    print("\n4. Testing Warmup Scheduler")
    optimizer = get_optimizer(model.parameters(), learning_rate=0.01)
    main_scheduler = get_scheduler(optimizer, 'step', step_size=30)
    warmup = WarmupScheduler(optimizer, warmup_epochs=5, after_scheduler=main_scheduler)
    
    print("   LR schedule:")
    for epoch in range(10):
        lr = optimizer.param_groups[0]['lr']
        print(f"     Epoch {epoch}: {lr:.6f}")
        warmup.step()
    
    # Test 5: Scheduler stepping
    print("\n5. Testing Scheduler Step")
    optimizer = get_optimizer(model.parameters(), learning_rate=0.01)
    scheduler = get_scheduler(optimizer, 'exponential', gamma=0.9)
    
    print("   LR decay:")
    for epoch in range(5):
        lr = optimizer.param_groups[0]['lr']
        print(f"     Epoch {epoch}: {lr:.6f}")
        if scheduler is not None:
            scheduler.step()
    
    print("\n Optimization tests complete!")




