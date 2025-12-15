"""
Visualization utilities for molecular property prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Union
from sklearn.metrics import confusion_matrix

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    history: dict,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Path to save figure (optional)
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics(
    history: dict,
    metrics: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot training metrics over epochs.
    
    Args:
        history: Dictionary with metric histories
        metrics: List of metric names to plot
        save_path: Path to save figure
        show: Whether to show the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        epochs = range(1, len(history.get(train_key, [])) + 1)
        
        if train_key in history:
            train_values = [m.get(metric, 0) if isinstance(m, dict) else m 
                           for m in history.get('train_metrics', [])]
            if train_values:
                ax.plot(epochs, train_values, 'b-', label='Train', linewidth=2)
        
        if val_key in history:
            val_values = [m.get(metric, 0) if isinstance(m, dict) else m 
                         for m in history.get('val_metrics', [])]
            if val_values:
                ax.plot(epochs, val_values, 'r-', label='Val', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save figure
        show: Whether to show the plot
    """

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes or ['Negative', 'Positive'],
        yticklabels=classes or ['Negative', 'Positive'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
        show: Whether to show the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plotting 
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'regression',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot prediction vs true value distribution.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        save_path: Path to save figure
        show: Whether to show the plot
    """
    if task_type == 'regression':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', linewidth=2)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predicted vs True Values')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
    else:  # classification
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram of predictions by class
        for label in np.unique(y_true):
            mask = y_true == label
            ax.hist(y_pred[mask], bins=50, alpha=0.5, 
                   label=f'Class {int(label)}')
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate_schedule(
    learning_rates: List[float],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates per epoch
        save_path: Path to save figure
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, 'b-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    results: dict,
    metric: str = 'roc_auc',
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot comparison of different models.
    
    Args:
        results: Dictionary mapping model names to metric values
        metric: Metric to compare
        save_path: Path to save figure
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    values = list(results.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison - {metric.upper()}')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    np.random.seed(42)
    
    print("\n1. Testing plot_training_curves()")
    history = {
        'train_loss': [0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25],
        'val_loss': [0.55, 0.48, 0.42, 0.38, 0.36, 0.35, 0.34]
    }
    plot_training_curves(history, show=False)
    print("    Training curves plotted")

    print("\n2. Testing plot_confusion_matrix()")
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    plot_confusion_matrix(y_true, y_pred, show=False)
    print("    Confusion matrix plotted")
    
    print("\n3. Testing plot_roc_curve()")
    y_true = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    plot_roc_curve(y_true, y_pred_proba, show=False)
    print("    ROC curve plotted")
    
    print("\n4. Testing plot_prediction_distribution() - regression")
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.3
    plot_prediction_distribution(y_true, y_pred, task_type='regression', show=False)
    print("    Regression prediction distribution plotted")
    

    print("\n5. Testing plot_prediction_distribution() - classification")
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.rand(100)
    plot_prediction_distribution(y_true, y_pred, task_type='classification', show=False)
    print("    Classification prediction distribution plotted")
    
    print("\n6. Testing plot_learning_rate_schedule()")
    lrs = [0.001 * (0.95 ** i) for i in range(50)]
    plot_learning_rate_schedule(lrs, show=False)
    print("    Learning rate schedule plotted")
    
    print("\n7. Testing plot_model_comparison()")
    results = {
        'GCN': 0.85,
        'GIN': 0.87,
        'GAT': 0.86,
        'Random Forest': 0.82,
        'MLP': 0.80
    }
    plot_model_comparison(results, show=False)
    print("    Model comparison plotted")
    
    print("\n All visualization tests complete!")
