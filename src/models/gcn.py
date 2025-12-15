"""
Graph Convolutional Network (GCN) for molecular property prediction.
Based on Kipf & Welling (2017) with modifications for molecular graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data
from .base_model import BaseMolecularModel
import os


class MolecularGCN(BaseMolecularModel):
    """
    Graph Convolutional Network for molecular property prediction.
    
    Architecture:
    - Multiple GCN layers with residual connections
    - Batch normalization for stable training
    - Global pooling to get graph-level representation
    - MLP head for final prediction
    
    GCN is simple but effective - good baseline model.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_tasks: int,
        num_layers: int = 5,
        dropout: float = 0.1,
        pooling: str = 'mean',
        use_residual: bool = True,
        use_batch_norm: bool = True
    ):
        """
        Initialize GCN model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features (not used in basic GCN)
            hidden_dim: Hidden dimension for graph convolutions
            num_tasks: Number of prediction tasks
            num_layers: Number of GCN layers
            dropout: Dropout probability
            pooling: Global pooling method ('mean', 'add', 'max')
            use_residual: Whether to use residual connections
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            num_layers=num_layers,
            pooling=pooling,
            use_residual=use_residual,
            use_batch_norm=use_batch_norm
        )
        
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Input layer
        self.convs.append(GCNConv(node_feat_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling layer
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Prediction head (MLP)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_tasks)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            predictions: [batch_size, num_tasks]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x_input = x
            
            x = conv(x, edge_index)
            
            # Batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer as dimensions differ)
            if self.use_residual and i > 0:
                x = x + x_input
        
        # Global pooling
        x = self.pool(x, batch)
        
        # MLP head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get graph-level embeddings before final prediction.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x_input = x
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if self.use_residual and i > 0:
                x = x + x_input
        
        x = self.pool(x, batch)
        
        return x



if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    print("Testing MolecularGCN...")
    
    # Create a sample molecule (benzene-like)
    x = torch.randn(6, 50)  # 6 atoms, 50 features
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)
    
    data1 = Data(x=x, edge_index=edge_index)
    data1.batch = torch.zeros(6, dtype=torch.long)
    
    # Create another molecule
    x2 = torch.randn(4, 50)
    edge_index2 = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    data2 = Data(x=x2, edge_index=edge_index2)
    data2.batch = torch.ones(4, dtype=torch.long)
    
    # Batch data
    batch_data = Batch.from_data_list([data1, data2])
    
    print(f"Batch: {batch_data.num_graphs} graphs, {batch_data.num_nodes} total nodes")
    
    model = MolecularGCN(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=128,
        num_tasks=1,
        num_layers=5,
        dropout=0.1,
        pooling='mean',
        use_residual=True,
        use_batch_norm=True
    )
    
    print(f"\n{model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [2, 1]")
    assert output.shape == (2, 1), f"Wrong output shape: {output.shape}"
    
    # Test embeddings
    with torch.no_grad():
        embeddings = model.get_embeddings(batch_data)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Expected: [2, 128]")
    assert embeddings.shape == (2, 128), f"Wrong embeddings shape: {embeddings.shape}"
    
    # Test different pooling methods
    print("\nTesting pooling methods:")
    for pooling in ['mean', 'add', 'max']:
        model = MolecularGCN(
            node_feat_dim=50,
            edge_feat_dim=10,
            hidden_dim=64,
            num_tasks=1,
            num_layers=3,
            pooling=pooling
        )
        
        with torch.no_grad():
            output = model(batch_data)
        
        print(f"  {pooling}: output shape = {output.shape}")
    
    # Test save/load
    print("\nTesting save/load:")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    model.save_checkpoint(temp_path, epoch=5, best_loss=0.3)
    loaded_model, checkpoint = MolecularGCN.load_checkpoint(temp_path)
    
    print(f"  Saved and loaded successfully")
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    

    os.unlink(temp_path)
    
    # Test gradient flow
    print("\nTesting gradient flow:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    output = model(batch_data)
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"  Gradients computed: {has_grads}")
    
    optimizer.step()
    print(f"  Optimizer step successful")
    
    print("\n GCN tests complete!")
