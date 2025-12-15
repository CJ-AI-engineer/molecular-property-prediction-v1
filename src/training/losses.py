"""
Loss functions for molecular property prediction.
Includes standard losses and specialized losses for molecular tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy loss with logits.
    For binary classification tasks.
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize BCE loss.
        
        Args:
            pos_weight: Weight for positive class (for imbalanced data)
        """
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions [batch_size, num_tasks]
            targets: Ground truth [batch_size, num_tasks]
            
        Returns:
            loss: Scalar loss value
        """
        return self.loss_fn(predictions, targets)


class MSELoss(nn.Module):
    """
    Mean Squared Error loss.
    For regression tasks.
    """
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return self.loss_fn(predictions, targets)


class MAELoss(nn.Module):
    """
    Mean Absolute Error loss (L1 loss).
    For regression tasks, more robust to outliers than MSE.
    """
    
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss."""
        return self.loss_fn(predictions, targets)


class HuberLoss(nn.Module):
    """
    Huber loss (smooth L1 loss).
    Robust to outliers while still being differentiable.
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold at which to change from quadratic to linear
        """
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss."""
        return self.loss_fn(predictions, targets)


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    
    Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions (logits) [batch_size, num_tasks]
            targets: Ground truth [batch_size, num_tasks]
            
        Returns:
            loss: Scalar loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)
        
        # Compute binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for handling sample importance.
    Useful when some predictions are more important than others.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: Model predictions [batch_size, num_tasks]
            targets: Ground truth [batch_size, num_tasks]
            weights: Sample weights [batch_size] or [batch_size, num_tasks]
            
        Returns:
            loss: Scalar loss value
        """
        mse = (predictions - targets) ** 2
        
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)
            mse = mse * weights
        
        return mse.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with automatic task weighting.
    Learns optimal weights for multiple tasks.
    
    Reference: Kendall et al. (2018) "Multi-Task Learning Using 
    Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(
        self,
        num_tasks: int,
        task_types: list,  
        learn_weights: bool = True
    ):
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks
            task_types: Type of each task ('classification' or 'regression')
            learn_weights: Whether to learn task weights
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        self.task_types = task_types
        self.learn_weights = learn_weights
        
        if learn_weights:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.log_vars = None
        
        # Individual loss functions
        self.loss_fns = []
        for task_type in task_types:
            if task_type == 'classification':
                self.loss_fns.append(nn.BCEWithLogitsLoss())
            elif task_type == 'regression':
                self.loss_fns.append(nn.MSELoss())
            else:
                raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions [batch_size, num_tasks]
            targets: Ground truth [batch_size, num_tasks]
            
        Returns:
            total_loss: Weighted sum of task losses
        """
        total_loss = 0
        
        for i, (loss_fn, task_type) in enumerate(zip(self.loss_fns, self.task_types)):
            task_pred = predictions[:, i:i+1]
            task_target = targets[:, i:i+1]
            task_loss = loss_fn(task_pred, task_target)
            
            if self.learn_weights:
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * task_loss + self.log_vars[i]
            else:
                total_loss += task_loss
        
        return total_loss / self.num_tasks


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss.
    Ensures that higher-affinity molecules are ranked higher.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize ranking loss.
        
        Args:
            margin: Margin for ranking constraint
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            predictions: Model predictions [batch_size]
            targets: Ground truth [batch_size]
            
        Returns:
            loss: Ranking loss
        """
        # Create all pairs
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        
        # Only consider pairs where target_diff > 0 (i > j in ranking)
        mask = (target_diff > 0).float()
        
        # Hinge loss: max(0, margin - (pred_i - pred_j))
        loss = F.relu(self.margin - pred_diff) * mask
        
        # Average over valid pairs
        num_pairs = mask.sum()
        if num_pairs > 0:
            return loss.sum() / num_pairs
        else:
            return torch.tensor(0.0, device=predictions.device)


def get_loss_fn(
    task_type: str,
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs
):
    """
    Factory function to get loss function by task type.
    
    Args:
        task_type: Type of task
        pos_weight: Positive class weight for BCE
        **kwargs: Additional loss-specific arguments
        
    Returns:
        Loss function
    """
    if task_type == 'classification' or task_type == 'binary':
        return BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif task_type == 'regression':
        loss_type = kwargs.get('loss_type', 'mse')
        if loss_type == 'mse':
            return MSELoss()
        elif loss_type == 'mae':
            return MAELoss()
        elif loss_type == 'huber':
            return HuberLoss(delta=kwargs.get('delta', 1.0))
        else:
            raise ValueError(f"Unknown regression loss: {loss_type}")
    
    elif task_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 0.25),
            gamma=kwargs.get('gamma', 2.0)
        )
    
    elif task_type == 'ranking':
        return RankingLoss(margin=kwargs.get('margin', 1.0))
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")



if __name__ == "__main__":
    print("Testing loss functions...")
    
    batch_size = 32
    num_tasks = 1
    
    # Test 1: BCE Loss
    print("\n1. Testing BCE Loss")
    bce_loss = BCEWithLogitsLoss()
    
    predictions = torch.randn(batch_size, num_tasks)
    targets = torch.randint(0, 2, (batch_size, num_tasks)).float()
    
    loss = bce_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test with class imbalance
    pos_weight = torch.tensor([2.0]) 
    bce_weighted = BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_weighted = bce_weighted(predictions, targets)
    print(f"   Weighted loss: {loss_weighted.item():.4f}")
    
    # Test 2: MSE Loss
    print("\n2. Testing MSE Loss")
    mse_loss = MSELoss()
    
    predictions = torch.randn(batch_size, num_tasks)
    targets = torch.randn(batch_size, num_tasks)
    
    loss = mse_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 3: MAE Loss
    print("\n3. Testing MAE Loss")
    mae_loss = MAELoss()
    loss = mae_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 4: Focal Loss
    print("\n4. Testing Focal Loss")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    predictions = torch.randn(batch_size, num_tasks)
    targets = torch.randint(0, 2, (batch_size, num_tasks)).float()
    
    loss = focal_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 5: Multi-task Loss
    print("\n5. Testing Multi-Task Loss")
    num_tasks = 3
    task_types = ['classification', 'regression', 'classification']
    
    mt_loss = MultiTaskLoss(
        num_tasks=num_tasks,
        task_types=task_types,
        learn_weights=True
    )
    
    predictions = torch.randn(batch_size, num_tasks)
    targets = torch.randn(batch_size, num_tasks)
    targets[:, [0, 2]] = torch.randint(0, 2, (batch_size, 2)).float()
    
    loss = mt_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Learned weights: {mt_loss.log_vars.data}")
    
    # Test 6: Ranking Loss
    print("\n6. Testing Ranking Loss")
    ranking_loss = RankingLoss(margin=1.0)
    
    predictions = torch.randn(batch_size)
    targets = torch.randn(batch_size)
    
    loss = ranking_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test 7: Factory function
    print("\n7. Testing Factory Function")
    for task_type in ['classification', 'regression', 'focal']:
        loss_fn = get_loss_fn(task_type)
        print(f"   {task_type}: {type(loss_fn).__name__}")
    
    # Test gradient flow
    print("\n8. Testing Gradient Flow")
    loss_fn = BCEWithLogitsLoss()
    predictions = torch.randn(batch_size, num_tasks, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, num_tasks)).float()
    
    loss = loss_fn(predictions, targets)
    loss.backward()
    
    print(f"Gradients computed: {predictions.grad is not None}")
    print(f"Gradient shape: {predictions.grad.shape}")
    
    print("\n Loss function tests complete!")
