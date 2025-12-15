"""
Base model class for molecular property prediction.
All GNN models inherit from this to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import os
import tempfile


class BaseMolecularModel(ABC, nn.Module):
    """
    Abstract base class for all molecular property prediction models.
    
    Ensures consistent interface across GCN, GIN, GAT, etc.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_tasks: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize base model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            hidden_dim: Hidden dimension for graph convolutions
            num_tasks: Number of prediction tasks
            dropout: Dropout probability
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.dropout = dropout
        
        self.model_config = kwargs
    
    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_feat_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feat_dim]
                - batch: Batch assignment [num_nodes]
                
        Returns:
            predictions: [batch_size, num_tasks]
        """
        raise NotImplementedError
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get graph-level embeddings before final prediction layer.
        Useful for visualization and transfer learning.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        # Default implementation - can be overridden
        # This runs forward but returns embeddings instead of predictions
        raise NotImplementedError(
            "get_embeddings not implemented for this model. "
            "Override this method in subclass if needed."
        )
    
    def count_parameters(self) -> int:
        """
        Count trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_parameters(self):
        """
        Reset all model parameters.
        Useful for retraining or cross-validation.
        """
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as dictionary.
        Useful for saving/loading and experiment tracking.
        
        Returns:
            Dictionary of model configuration
        """
        config = {
            'model_class': self.__class__.__name__,
            'node_feat_dim': self.node_feat_dim,
            'edge_feat_dim': self.edge_feat_dim,
            'hidden_dim': self.hidden_dim,
            'num_tasks': self.num_tasks,
            'dropout': self.dropout,
        }
        config.update(self.model_config)
        return config
    
    def save_checkpoint(self, path: str, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional items to save (optimizer state, epoch, etc.)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config(),
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, map_location: Optional[str] = None):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            map_location: Device to load model to
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['model_config']
        config.pop('model_class', None)
        
        model = cls(**config)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def __repr__(self):
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  node_feat_dim={self.node_feat_dim},\n"
            f"  edge_feat_dim={self.edge_feat_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_tasks={self.num_tasks},\n"
            f"  dropout={self.dropout},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


class BaselineModel(nn.Module):
    """
    Base class for non-GNN baseline models.
    For models using classical molecular descriptors.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_tasks: int,
        dropout: float = 0.1
    ):
        """
        Initialize baseline model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_tasks: Number of prediction tasks
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_tasks = num_tasks
        self.dropout = dropout
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_tasks))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            predictions: [batch_size, num_tasks]
        """
        return self.mlp(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing BaseMolecularModel...")
    
    class DummyModel(BaseMolecularModel):
        def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_tasks, dropout=0.1):
            super().__init__(node_feat_dim, edge_feat_dim, hidden_dim, num_tasks, dropout)
            
            self.fc = nn.Linear(hidden_dim, num_tasks)
        
        def forward(self, data):
            batch_size = data.batch.max().item() + 1
            return torch.randn(batch_size, self.num_tasks)
    
    model = DummyModel(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=128,
        num_tasks=1,
        dropout=0.1
    )
    
    print(f"\n{model}")
    
    config = model.get_config()
    print(f"\nConfig: {config}")
    
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    model.save_checkpoint(temp_path, epoch=10, best_loss=0.5)
    print(f"\n Saved checkpoint to {temp_path}")
    
    loaded_model, checkpoint = DummyModel.load_checkpoint(temp_path)
    print(f" Loaded checkpoint (epoch: {checkpoint.get('epoch')})")
    
 
    os.unlink(temp_path)
    
    print("\n\nTesting BaselineModel...")
    baseline = BaselineModel(
        input_dim=2048,
        hidden_dims=[512, 256, 128],
        num_tasks=1,
        dropout=0.1
    )
    
    print(f"Parameters: {baseline.count_parameters():,}")
    
    x = torch.randn(32, 2048)
    out = baseline(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    print("\n Base model tests complete!")
