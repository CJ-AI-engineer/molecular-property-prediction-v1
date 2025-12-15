"""
Training module for molecular property prediction.

Provides:
- Training loop (Trainer)
- Loss functions
- Evaluation metrics  
- Optimizers and schedulers
- Training callbacks
"""

from .trainer import MolecularPropertyTrainer
from .losses import (
    BCEWithLogitsLoss,
    MSELoss,
    MAELoss,
    HuberLoss,
    FocalLoss,
    WeightedMSELoss,
    MultiTaskLoss,
    RankingLoss,
    get_loss_fn
)
from .metrics import (
    compute_roc_auc,
    compute_pr_auc,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_mse,
    MetricCalculator
)
from .optimization import (
    get_optimizer,
    get_scheduler,
    configure_optimization,
    WarmupScheduler
)
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateLogger,
    GradientLogger,
    MetricLogger,
    ProgressCallback
)

__all__ = [
    # Trainer
    'MolecularPropertyTrainer',
    
    # Losses
    'BCEWithLogitsLoss',
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'FocalLoss',
    'WeightedMSELoss',
    'MultiTaskLoss',
    'RankingLoss',
    'get_loss_fn',
    
    # Metrics
    'compute_roc_auc',
    'compute_pr_auc',
    'compute_accuracy',
    'compute_precision_recall_f1',
    'compute_rmse',
    'compute_mae',
    'compute_r2',
    'compute_mse',
    'MetricCalculator',
    
    # Optimization
    'get_optimizer',
    'get_scheduler',
    'configure_optimization',
    'WarmupScheduler',
    
    # Callbacks
    'Callback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateLogger',
    'GradientLogger',
    'MetricLogger',
    'ProgressCallback',
]
