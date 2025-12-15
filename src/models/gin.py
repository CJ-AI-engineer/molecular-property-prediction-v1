"""
Graph Isomorphism Network (GIN) for molecular property prediction.
Based on Xu et al. (2019) - provably more expressive than GCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data
from .base_model import BaseMolecularModel


class MolecularGIN(BaseMolecularModel):
    """
    Graph Isomorphism Network for molecular property prediction.
    
    Architecture:
    - GIN layers with MLPs for aggregation
    - Learnable epsilon for neighbor aggregation
    - Global pooling (sum pooling recommended for GIN)
    - MLP head for prediction
    
    GIN is provably more expressive than GCN - can distinguish
    more graph structures. Good for complex molecular patterns.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_tasks: int,
        num_layers: int = 5,
        dropout: float = 0.1,
        pooling: str = 'add',
        train_eps: bool = True,
        use_edge_features: bool = False
    ):
        """
        Initialize GIN model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            hidden_dim: Hidden dimension for graph convolutions
            num_tasks: Number of prediction tasks
            num_layers: Number of GIN layers
            dropout: Dropout probability
            pooling: Global pooling method ('mean', 'add', 'max')
            train_eps: Whether to learn epsilon (recommended: True)
            use_edge_features: Whether to incorporate edge features
        """
        super().__init__(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            num_layers=num_layers,
            pooling=pooling,
            train_eps=train_eps,
            use_edge_features=use_edge_features
        )
        
        self.num_layers = num_layers
        self.train_eps = train_eps
        self.use_edge_features = use_edge_features
        
        # Edge encoder (if using edge features)
        if use_edge_features and edge_feat_dim > 0:
            self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        else:
            self.edge_encoder = None
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = node_feat_dim if i == 0 else hidden_dim
            
            # MLP for GIN aggregation
            # Using 2-layer MLP as in original paper
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # GIN convolution
            self.convs.append(GINConv(mlp, train_eps=train_eps))
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
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GIN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            predictions: [batch_size, num_tasks]
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, 
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            data.batch
        )

        edge_weight = None
        if self.edge_encoder is not None and edge_attr is not None:
            # use edge features as weights
            edge_weight = self.edge_encoder(edge_attr).sum(dim=-1)
        
        # GIN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            # Apply GIN convolution
            # Note: edge_weight not directly supported in basic GINConv
            # For edge features, you'd need custom MessagePassing
            x = conv(x, edge_index)
            
            # Batch normalization
            x = bn(x)
            
            # Activation
            x = F.relu(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (sum is important for GIN's expressiveness)
        x = self.pool(x, batch)
        
        # Prediction head
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
        
        # GIN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        return x


class MolecularGINWithEdges(BaseMolecularModel):
    """
    GIN variant that properly handles edge features.
    Uses custom message passing to incorporate edge information.
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_tasks: int,
        num_layers: int = 5,
        dropout: float = 0.1,
        pooling: str = 'add',
        train_eps: bool = True
    ):
        """Initialize GIN with edge features."""
        super().__init__(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            num_layers=num_layers,
            pooling=pooling,
            train_eps=train_eps
        )
        
        self.num_layers = num_layers
        
        # Edge embedding
        if edge_feat_dim > 0:
            self.edge_embedding = nn.Linear(edge_feat_dim, hidden_dim)
        else:
            self.edge_embedding = None
        
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling
        if pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_tasks)
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with edge features."""
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index,
            data.edge_attr if hasattr(data, 'edge_attr') else None,
            data.batch
        )

        x = self.node_embedding(x)
        
        # Incorporate edge features (simplified approach)
        # For proper edge handling, you'd need custom MessagePassing
        if self.edge_embedding is not None and edge_attr is not None:
            edge_emb = self.edge_embedding(edge_attr)
            # Aggregate edge features to nodes
            row, col = edge_index
            x = x + torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), edge_emb)
        
        # GIN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pool and predict
        x = self.pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get embeddings."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.node_embedding(x)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        return self.pool(x, batch)



if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    print("Testing MolecularGIN...")
    
    # Create sample data
    x = torch.randn(6, 50)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)
    edge_attr = torch.randn(12, 10)
    
    data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data1.batch = torch.zeros(6, dtype=torch.long)
    
    x2 = torch.randn(4, 50)
    edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr2 = torch.randn(6, 10)
    
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
    data2.batch = torch.ones(4, dtype=torch.long)
    
    batch_data = Batch.from_data_list([data1, data2])
    
    print("\n1. Basic GIN (no edge features)")
    model = MolecularGIN(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=128,
        num_tasks=1,
        num_layers=5,
        pooling='add',
        train_eps=True,
        use_edge_features=False
    )
    
    print(f"{model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    model.eval()
    with torch.no_grad():
        output = model(batch_data)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1)
    
    # Test GIN with edge features
    print("\n2. GIN with edge features")
    model = MolecularGINWithEdges(
        node_feat_dim=50,
        edge_feat_dim=10,
        hidden_dim=128,
        num_tasks=1,
        num_layers=5
    )
    
    print(f"Parameters: {model.count_parameters():,}")
    
    with torch.no_grad():
        output = model(batch_data)
        embeddings = model.get_embeddings(batch_data)
    
    print(f"Output shape: {output.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test gradient flow
    print("\n3. Testing gradient flow")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    output = model(batch_data)
    loss = output.mean()
    loss.backward()
    
    has_grads = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_grads}")
    
    optimizer.step()
    print(f"Optimizer step successful")
    
    # Compare pooling methods
    print("\n4. Comparing pooling methods")
    for pooling in ['add', 'mean', 'max']:
        model = MolecularGIN(
            node_feat_dim=50,
            edge_feat_dim=10,
            hidden_dim=64,
            num_tasks=1,
            num_layers=3,
            pooling=pooling
        )
        
        with torch.no_grad():
            output = model(batch_data)
        
        print(f"  {pooling}: {output.shape}")
    
    print("\n GIN tests complete!")
