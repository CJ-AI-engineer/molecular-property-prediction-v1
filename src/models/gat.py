"""
Graph Attention Network (GAT) for molecular property prediction.
Uses attention mechanism to weight neighbor contributions.
Good for interpretability - can visualize which bonds/atoms are important.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data
from typing import Optional, Tuple

from .base_model import BaseMolecularModel


class MolecularGAT(BaseMolecularModel):
    """
    Graph Attention Network for molecular property prediction.
    
    Architecture:
    - Multi-head GAT layers
    - Attention weights for edge importance
    - Global pooling
    - MLP head for prediction
    
    GAT learns to focus on important neighbors - excellent for
    interpretability and understanding which molecular substructures matter.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_tasks: int,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: str = 'mean',
        concat_heads: bool = True,
        use_edge_features: bool = True
    ):
        """
        Initialize GAT model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            hidden_dim: Hidden dimension (per attention head)
            num_tasks: Number of prediction tasks
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability (also used for attention dropout)
            pooling: Global pooling method ('mean', 'add', 'max')
            concat_heads: Whether to concatenate or average attention heads
            use_edge_features: Whether to use edge features in attention
        """
        super().__init__(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            pooling=pooling,
            concat_heads=concat_heads,
            use_edge_features=use_edge_features
        )
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.use_edge_features = use_edge_features
        
        # Calculate dimensions
        # When concatenating heads: out_dim = hidden_dim * num_heads
        # When averaging heads: out_dim = hidden_dim
        head_dim = hidden_dim // num_heads if concat_heads else hidden_dim
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(
            GATConv(
                in_channels=node_feat_dim,
                out_channels=head_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_feat_dim if use_edge_features and edge_feat_dim > 0 else None,
                concat=concat_heads
            )
        )
        
        # Hidden layers
        in_dim = hidden_dim if concat_heads else head_dim
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_feat_dim if use_edge_features and edge_feat_dim > 0 else None,
                    concat=concat_heads
                )
            )
        
        # Pooling layer
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Prediction head
        final_dim = hidden_dim if concat_heads else head_dim
        self.fc1 = nn.Linear(final_dim, final_dim // 2)
        self.fc2 = nn.Linear(final_dim // 2, num_tasks)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(
        self,
        data: Data,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through GAT.
        
        Args:
            data: PyTorch Geometric Data object
            return_attention: Whether to return attention weights
            
        Returns:
            predictions: [batch_size, num_tasks]
            attention_weights (optional): Attention weights from last layer
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            data.batch
        )
        
        self.attention_weights = None
        
        # GAT layers
        for i, conv in enumerate(self.convs):
            if return_attention and i == len(self.convs) - 1:
                x, (edge_idx, attn) = conv(
                    x, edge_index,
                    edge_attr=edge_attr if self.use_edge_features else None,
                    return_attention_weights=True
                )
                self.attention_weights = (edge_idx, attn)
            else:
                x = conv(
                    x, edge_index,
                    edge_attr=edge_attr if self.use_edge_features else None
                )
            
            x = F.elu(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        
        # Prediction head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        if return_attention:
            return x, self.attention_weights
        return x
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get graph-level embeddings before final prediction.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            data.batch
        )
        
        # GAT layers
        for conv in self.convs:
            x = conv(
                x, edge_index,
                edge_attr=edge_attr if self.use_edge_features else None
            )
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        return x
    
    def get_attention_weights(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the last stored attention weights.
        Call forward with return_attention=True first.
        
        Returns:
            Tuple of (edge_index, attention_weights) or None
        """
        return self.attention_weights



if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    print("Testing MolecularGAT...")
    
    # Create sample data (benzene-like structure or benzene ring)
    x = torch.randn(6, 50)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)
    edge_attr = torch.randn(12, 10)
    
    data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data1.batch = torch.zeros(6, dtype=torch.long)
    
    # Second molecule
    x2 = torch.randn(4, 50)
    edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr2 = torch.randn(6, 10)
    
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
    data2.batch = torch.ones(4, dtype=torch.long)
    
    batch_data = Batch.from_data_list([data1, data2])
    
    print(f"Batch: {batch_data.num_graphs} graphs, {batch_data.num_nodes} nodes")
    
    # Test 1: Basic GAT
    print("\n1. Testing basic GAT")
    model = MolecularGAT(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=128,
        num_tasks=1,
        num_layers=5,
        num_heads=4,
        dropout=0.1,
        pooling='mean',
        concat_heads=True,
        use_edge_features=True
    )
    
    print(f"{model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1), f"Wrong output shape: {output.shape}"
    
    # Test 2: Attention weights
    print("\n2. Testing attention visualization")
    with torch.no_grad():
        output, attn_weights = model(batch_data, return_attention=True)
    
    if attn_weights is not None:
        edge_idx, attn = attn_weights
        print(f"Attention edge_index shape: {edge_idx.shape}")
        print(f"Attention weights shape: {attn.shape}")
        print(f"Attention heads: {attn.shape[1] if len(attn.shape) > 1 else 1}")
    
    # Test 3: Embeddings
    print("\n3. Testing embeddings")
    with torch.no_grad():
        embeddings = model.get_embeddings(batch_data)
    
    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (2, 128), f"Wrong embeddings shape: {embeddings.shape}"
    
    # Test 4: Different configurations
    print("\n4. Testing different configurations")
    
    # Average heads instead of concat
    model_avg = MolecularGAT(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=64,
        num_tasks=1,
        num_layers=3,
        num_heads=4,
        concat_heads=False
    )
    
    with torch.no_grad():
        output = model_avg(batch_data)
    print(f"Average heads - Output: {output.shape}")
    
    # Without edge features
    model_no_edge = MolecularGAT(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=64,
        num_tasks=1,
        num_layers=3,
        num_heads=4,
        use_edge_features=False
    )
    
    with torch.no_grad():
        output = model_no_edge(batch_data)
    print(f"No edge features - Output: {output.shape}")
    
    # Test 5: Different pooling methods
    print("\n5. Testing pooling methods")
    for pooling in ['mean', 'add', 'max']:
        model = MolecularGAT(
            node_feat_dim=50,
            edge_feat_dim=10,
            hidden_dim=64,
            num_tasks=1,
            num_layers=3,
            num_heads=4,
            pooling=pooling
        )
        
        with torch.no_grad():
            output = model(batch_data)
        
        print(f"  {pooling}: {output.shape}")
    
    # Test 6: Gradient flow
    print("\n6. Testing gradient flow")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    output = model(batch_data)
    loss = output.mean()
    loss.backward()
    
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_grads}")
    
    optimizer.step()
    print(f"Optimizer step successful")
    
    # Test 7: Save/load
    print("\n7. Testing save/load")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    model.save_checkpoint(temp_path, epoch=10, best_loss=0.25)
    loaded_model, checkpoint = MolecularGAT.load_checkpoint(temp_path)
    
    print(f"Saved and loaded successfully")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    import os
    os.unlink(temp_path)
    
    print("\n GAT tests complete!")
