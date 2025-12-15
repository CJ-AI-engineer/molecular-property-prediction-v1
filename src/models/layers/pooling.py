"""
Custom pooling layers for graph neural networks.
Provides different ways to aggregate node features to graph-level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import Set2Set, GlobalAttention


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling that combines mean, max, and sum pooling.
    """
    
    def __init__(self, mode: str = 'concat'):
        """
        Initialize adaptive pooling.
        
        Args:
            mode: How to combine pooled features
                - 'concat': Concatenate mean, max, sum
                - 'mean': Average mean, max, sum
                - 'weighted': Learned weighted combination
        """
        super().__init__()
        self.mode = mode
        
        if mode == 'weighted':
            self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            pooled: Graph-level features
        """

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        
        if self.mode == 'concat':
            return torch.cat([mean_pool, max_pool, sum_pool], dim=-1)
        
        elif self.mode == 'mean':
            return (mean_pool + max_pool + sum_pool) / 3
        
        elif self.mode == 'weighted':
            w = F.softmax(self.weights, dim=0)
            return w[0] * mean_pool + w[1] * max_pool + w[2] * sum_pool
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def output_dim_multiplier(self):
        """Return dimension multiplier for output."""
        return 3 if self.mode == 'concat' else 1


class AttentionalPooling(nn.Module):
    """
    Attention-based pooling.
    Learns to weight nodes by importance.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize attentional pooling.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch):
        """
        Forward pass with attention.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            pooled: Graph-level features [batch_size, hidden_dim]
        """
        # Compute attention scores
        attn_scores = self.attention_net(x)  # [num_nodes, 1]
        
        # Use PyG's GlobalAttention
        pool = GlobalAttention(self.attention_net)
        return pool(x, batch)


class Set2SetPooling(nn.Module):
    """
    Set2Set pooling layer.
    Uses LSTM to process set of nodes.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        processing_steps: int = 3
    ):
        """
        Initialize Set2Set pooling.
        
        Args:
            hidden_dim: Hidden dimension
            processing_steps: Number of processing steps
        """
        super().__init__()
        
        self.set2set = Set2Set(
            in_channels=hidden_dim,
            processing_steps=processing_steps
        )
    
    def forward(self, x, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            pooled: Graph-level features [batch_size, 2 * hidden_dim]
        """
        return self.set2set(x, batch)
    
    def output_dim_multiplier(self):
        """Set2Set doubles the dimension."""
        return 2


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling that combines local and global features.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize hierarchical pooling.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.local_pool = global_max_pool
        self.global_pool = global_mean_pool
        
        # Combine local and global for molecules 
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, batch):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            pooled: Graph-level features [batch_size, hidden_dim]
        """
        local = self.local_pool(x, batch)
        global_ = self.global_pool(x, batch)
        
        combined = torch.cat([local, global_], dim=-1)
        return self.combine(combined)


def get_pooling_layer(
    pooling_type: str,
    hidden_dim: int = None,
    **kwargs
):
    """
    Factory function to get pooling layer.
    
    Args:
        pooling_type: Type of pooling
        hidden_dim: Hidden dimension (for some pooling types)
        **kwargs: Additional arguments
        
    Returns:
        Pooling layer
    """
    if pooling_type == 'mean':
        return global_mean_pool
    
    elif pooling_type == 'add' or pooling_type == 'sum':
        return global_add_pool
    
    elif pooling_type == 'max':
        return global_max_pool
    
    elif pooling_type == 'adaptive':
        mode = kwargs.get('mode', 'concat')
        return AdaptivePooling(mode=mode)
    
    elif pooling_type == 'attention':
        if hidden_dim is None:
            raise ValueError("hidden_dim required for attention pooling")
        return AttentionalPooling(hidden_dim=hidden_dim)
    
    elif pooling_type == 'set2set':
        if hidden_dim is None:
            raise ValueError("hidden_dim required for set2set pooling")
        steps = kwargs.get('processing_steps', 3)
        return Set2SetPooling(hidden_dim=hidden_dim, processing_steps=steps)
    
    elif pooling_type == 'hierarchical':
        if hidden_dim is None:
            raise ValueError("hidden_dim required for hierarchical pooling")
        return HierarchicalPooling(hidden_dim=hidden_dim)
    
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    
    print("Testing pooling layers...")
    
    # Create sample data
    x1 = torch.randn(6, 64)  # 6 nodes, 64 features
    x2 = torch.randn(4, 64)  # 4 nodes, 64 features
    
    batch = torch.cat([
        torch.zeros(6, dtype=torch.long),
        torch.ones(4, dtype=torch.long)
    ])
    
    x = torch.cat([x1, x2], dim=0)
    
    print(f"Input: {x.shape}, Batch size: {batch.max().item() + 1}")
    
    # Test 1: Adaptive pooling
    print("\n1. Adaptive Pooling")
    for mode in ['concat', 'mean', 'weighted']:
        pool = AdaptivePooling(mode=mode)
        out = pool(x, batch)
        print(f"  {mode}: {out.shape}")
    
    # Test 2: Attentional pooling
    print("\n2. Attentional Pooling")
    pool = AttentionalPooling(hidden_dim=64)
    out = pool(x, batch)
    print(f"  Output shape: {out.shape}")
    
    # Test 3: Set2Set pooling
    print("\n3. Set2Set Pooling")
    pool = Set2SetPooling(hidden_dim=64, processing_steps=3)
    out = pool(x, batch)
    print(f"  Output shape: {out.shape}")
    print(f"  Dimension multiplier: {pool.output_dim_multiplier()}")
    
    # Test 4: Hierarchical pooling
    print("\n4. Hierarchical Pooling")
    pool = HierarchicalPooling(hidden_dim=64)
    out = pool(x, batch)
    print(f"  Output shape: {out.shape}")
    
    # Test 5: Factory function
    print("\n5. Using Factory Function")
    for pooling_type in ['mean', 'add', 'max', 'adaptive', 'attention', 'set2set']:
        pool = get_pooling_layer(pooling_type, hidden_dim=64)
        
        # Handle different return types
        if callable(pool):
            if pooling_type in ['mean', 'add', 'max']:
                out = pool(x, batch)
            else:
                out = pool(x, batch)
        else:
            out = pool(x, batch)
        
        print(f"  {pooling_type}: {out.shape}")
    
    print("\n6. Testing Gradient Flow")
    pool = AdaptivePooling(mode='weighted')
    out = pool(x, batch)
    loss = out.mean()
    loss.backward()
    
    print(f"  Gradients computed: {pool.weights.grad is not None}")
    
    print("\n Pooling layer tests complete!")
