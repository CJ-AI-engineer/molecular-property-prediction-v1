"""
Graph Neural Network models for molecular property prediction.

This module provides:
- GNN architectures (GCN, GIN, GAT)
- Baseline models (Random Forest, SVM, MLP)
- Custom layers and pooling
- Deep ensembles for uncertainty
"""

from .base_model import BaseMolecularModel, BaselineModel
from .gcn import MolecularGCN
from .gin import MolecularGIN, MolecularGINWithEdges
from .gat import MolecularGAT
from .baselines import (
    MLPBaseline,
    RandomForestBaseline,
    SVMBaseline,
    LogisticRegressionBaseline,
    RidgeBaseline
)
from .ensemble import DeepEnsemble
from .layers import (
    AdaptivePooling,
    AttentionalPooling,
    Set2SetPooling,
    HierarchicalPooling,
    get_pooling_layer
)

__all__ = [
    # Base classes
    'BaseMolecularModel',
    'BaselineModel',
    
    # GNN models
    'MolecularGCN',
    'MolecularGIN',
    'MolecularGINWithEdges',
    'MolecularGAT',
    
    # Baseline models
    'MLPBaseline',
    'RandomForestBaseline',
    'SVMBaseline',
    'LogisticRegressionBaseline',
    'RidgeBaseline',
    
    # Ensemble
    'DeepEnsemble',
    
    # Layers
    'AdaptivePooling',
    'AttentionalPooling',
    'Set2SetPooling',
    'HierarchicalPooling',
    'get_pooling_layer',
]


def get_model(
    model_type: str,
    node_feat_dim: int,
    edge_feat_dim: int,
    hidden_dim: int,
    num_tasks: int,
    **kwargs
):
    """
    Factory function to get model by name.
    
    Args:
        model_type: Model type ('gcn', 'gin', 'gat')
        node_feat_dim: Node feature dimension
        edge_feat_dim: Edge feature dimension
        hidden_dim: Hidden dimension
        num_tasks: Number of tasks
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    models = {
        'gcn': MolecularGCN,
        'gin': MolecularGIN,
        'gin_edge': MolecularGINWithEdges,
        'gat': MolecularGAT,
    }
    
    if model_type not in models:
        raise ValueError(
            f"Unknown model: {model_type}. "
            f"Choose from {list(models.keys())}"
        )
    
    model_class = models[model_type]
    
    return model_class(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_tasks=num_tasks,
        **kwargs
    )
