"""
Evaluation metrics for molecular property prediction.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Optional, Union


def compute_roc_auc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    average: str = 'macro'
) -> float:
    """
    Compute ROC-AUC score.
    
    Args:
        y_true: Ground truth labels [n_samples] or [n_samples, n_tasks]
        y_pred: Predicted probabilities [n_samples] or [n_samples, n_tasks]
        average: Averaging method for multi-task ('macro', 'micro', 'weighted')
        
    Returns:
        ROC-AUC score
    """
    # Convert tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Handle single task
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Compute ROC-AUC for each task
    scores = []
    for i in range(y_true.shape[1]):
        # Skip if only one class present
        if len(np.unique(y_true[:, i])) < 2:
            continue
        
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except:
            continue
    
    if len(scores) == 0:
        return 0.0
    
    return np.mean(scores)


def compute_pr_auc(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Precision-Recall AUC score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        PR-AUC score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        
        try:
            score = average_precision_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except:
            continue
    
    if len(scores) == 0:
        return 0.0
    
    return np.mean(scores)


def compute_accuracy(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Accuracy score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    return accuracy_score(y_true.flatten(), y_pred_binary.flatten())


def compute_precision_recall_f1(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with precision, recall, f1
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    return {
        'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_flat, zero_division=0)
    }


def compute_rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return mean_absolute_error(y_true, y_pred)


def compute_r2(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute R² score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return r2_score(y_true, y_pred)


def compute_mse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MSE
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return mean_squared_error(y_true, y_pred)


class MetricCalculator:
    """
    Unified metric calculator for both classification and regression.
    """
    
    def __init__(self, task_type: str = 'classification', threshold: float = 0.5):
        """
        Initialize metric calculator.
        
        Args:
            task_type: 'classification' or 'regression'
            threshold: Classification threshold
        """
        self.task_type = task_type
        self.threshold = threshold
    
    def compute_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_pred_proba: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Compute all relevant metrics.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions (probabilities for classification)
            y_pred_proba: Predicted probabilities (for classification, if y_pred is binary)
            
        Returns:
            Dictionary of metrics
        """
        if self.task_type == 'classification':
            return self._compute_classification_metrics(y_true, y_pred, y_pred_proba)
        else:
            return self._compute_regression_metrics(y_true, y_pred)
    
    def _compute_classification_metrics(
        self,
        y_true,
        y_pred,
        y_pred_proba=None
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        if y_pred_proba is None:
            y_pred_proba = y_pred
        
        metrics = {}
        
        # ROC-AUC and PR-AUC (need probabilities)
        metrics['roc_auc'] = compute_roc_auc(y_true, y_pred_proba)
        metrics['pr_auc'] = compute_pr_auc(y_true, y_pred_proba)
        
        # Accuracy, precision, recall, F1
        metrics['accuracy'] = compute_accuracy(y_true, y_pred_proba, self.threshold)
        prf = compute_precision_recall_f1(y_true, y_pred_proba, self.threshold)
        metrics.update(prf)
        
        return metrics
    
    def _compute_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Compute regression metrics."""
        
        metrics = {
            'rmse': compute_rmse(y_true, y_pred),
            'mae': compute_mae(y_true, y_pred),
            'mse': compute_mse(y_true, y_pred),
            'r2': compute_r2(y_true, y_pred)
        }
        
        return metrics


if __name__ == "__main__":
    print("Testing metric functions...")
    
    n_samples = 100
    
    # Test 1: Classification metrics
    print("\n1. Testing Classification Metrics")
    
    y_true_class = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.random.rand(n_samples)
    
    roc_auc = compute_roc_auc(y_true_class, y_pred_proba)
    pr_auc = compute_pr_auc(y_true_class, y_pred_proba)
    acc = compute_accuracy(y_true_class, y_pred_proba)
    prf = compute_precision_recall_f1(y_true_class, y_pred_proba)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prf['precision']:.4f}")
    print(f"Recall: {prf['recall']:.4f}")
    print(f"F1: {prf['f1']:.4f}")
    
    # Test 2: Regression metrics
    print("\n2. Testing Regression Metrics")
    
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.5
    
    rmse = compute_rmse(y_true_reg, y_pred_reg)
    mae = compute_mae(y_true_reg, y_pred_reg)
    r2 = compute_r2(y_true_reg, y_pred_reg)
    mse = compute_mse(y_true_reg, y_pred_reg)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Test 3: Multi-task classification
    print("\n3. Testing Multi-Task Classification")
    
    y_true_mt = np.random.randint(0, 2, (n_samples, 3))
    y_pred_mt = np.random.rand(n_samples, 3)
    
    roc_auc_mt = compute_roc_auc(y_true_mt, y_pred_mt)
    print(f"Multi-task ROC-AUC: {roc_auc_mt:.4f}")
    
    # Test 4: MetricCalculator
    print("\n4. Testing MetricCalculator")
    
    # Classification
    calc_class = MetricCalculator(task_type='classification')
    metrics_class = calc_class.compute_metrics(y_true_class, y_pred_proba)
    
    print(f"Classification metrics:")
    for name, value in metrics_class.items():
        print(f"     {name}: {value:.4f}")
    
    # Regression
    calc_reg = MetricCalculator(task_type='regression')
    metrics_reg = calc_reg.compute_metrics(y_true_reg, y_pred_reg)
    
    print(f"Regression metrics:")
    for name, value in metrics_reg.items():
        print(f"     {name}: {value:.4f}")
    
    # Test 5: PyTorch tensors
    print("\n5. Testing with PyTorch Tensors")
    
    y_true_torch = torch.randint(0, 2, (n_samples,)).float()
    y_pred_torch = torch.rand(n_samples)
    
    roc_auc_torch = compute_roc_auc(y_true_torch, y_pred_torch)
    acc_torch = compute_accuracy(y_true_torch, y_pred_torch)
    
    print(f"ROC-AUC (torch): {roc_auc_torch:.4f}")
    print(f"Accuracy (torch): {acc_torch:.4f}")
    
    print("\n Metric tests complete!")
