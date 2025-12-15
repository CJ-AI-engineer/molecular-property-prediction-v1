"""
Data augmentation techniques for molecular graphs. To improve model generalization and robustness.
"""

import torch
import numpy as np
from copy import deepcopy
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, dropout_node


class GraphAugmentation:
    """Base class for graph augmentation."""
    
    def __call__(self, data: Data) -> Data:
        """Apply augmentation to a graph."""
        raise NotImplementedError


class DropNodes(GraphAugmentation):
    """
    Randomly drop nodes from the graph. Use it for regularization and robustness.
    """
    
    def __init__(self, drop_prob: float = 0.1):
        """
        Initialize node dropout.
        
        Args:
            drop_prob: Probability of dropping each node
        """
        self.drop_prob = drop_prob
    
    def __call__(self, data: Data) -> Data:
        """
        Apply node dropout.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph with some nodes removed
        """
        if self.drop_prob == 0:
            return data
        
        data = deepcopy(data)
        
        # node 
        edge_index, edge_mask = dropout_node(
            data.edge_index,
            p=self.drop_prob,
            num_nodes=data.num_nodes
        )
        
        # edge 
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]
        
        data.edge_index = edge_index
        
        return data


class DropEdges(GraphAugmentation):
    """
    Randomly drop edges from the graph.
    """
    
    def __init__(self, drop_prob: float = 0.1):
        """
        Initialize edge dropout.
        
        Args:
            drop_prob: Probability of dropping each edge
        """
        self.drop_prob = drop_prob
    
    def __call__(self, data: Data) -> Data:
        """
        Apply edge dropout.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph with some edges removed
        """
        if self.drop_prob == 0:
            return data
        
        data = deepcopy(data)
        
        edge_index, edge_attr = dropout_adj(
            data.edge_index,
            edge_attr=data.edge_attr,
            p=self.drop_prob
        )
        
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        
        return data


class MaskNodeFeatures(GraphAugmentation):
    """
    Randomly mask node features.
    """
    
    def __init__(self, mask_prob: float = 0.15):
        """
        Initialize feature masking.
        
        Args:
            mask_prob: Probability of masking each feature
        """
        self.mask_prob = mask_prob
    
    def __call__(self, data: Data) -> Data:
        """
        Apply feature masking.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph with masked features
        """
        if self.mask_prob == 0:
            return data
        
        data = deepcopy(data)
        
        if data.x is not None:
            mask = torch.rand(data.x.shape) < self.mask_prob
            
            data.x = data.x.clone()
            data.x[mask] = 0
        
        return data


class AddGaussianNoise(GraphAugmentation):
    """
    Add Gaussian noise to node features.
    Improves robustness to input perturbations.
    """
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        """
        Initialize Gaussian noise addition.
        
        Args:
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, data: Data) -> Data:
        """
        Add Gaussian noise to features.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph with noisy features
        """
        if self.std == 0:
            return data
        
        data = deepcopy(data)
        
        if data.x is not None:
            noise = torch.randn_like(data.x) * self.std + self.mean
            data.x = data.x + noise
        
        return data


class RandomAttributeNoise(GraphAugmentation):
    """
    Randomly perturb categorical node/edge attributes.
    Specifically designed for molecular graphs with one-hot features.
    """
    
    def __init__(self, perturb_prob: float = 0.05):
        """
        Initialize attribute perturbation.
        
        Args:
            perturb_prob: Probability of perturbing each attribute
        """
        self.perturb_prob = perturb_prob
    
    def __call__(self, data: Data) -> Data:
        """
        Perturb categorical attributes.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph with perturbed attributes
        """
        if self.perturb_prob == 0:
            return data
        
        data = deepcopy(data)
        
        if data.x is not None:
            # Randomly flip some features for one-hot encodings
            mask = torch.rand(data.x.shape) < self.perturb_prob
            data.x = data.x.clone()
            data.x[mask] = 1 - data.x[mask]  # Flip binary features
        
        return data


class Compose:
    """
    Compose multiple augmentations together.
    Applies them sequentially.
    """
    
    def __init__(self, augmentations: list):
        """
        Initialize composition.
        
        Args:
            augmentations: List of GraphAugmentation instances
        """
        self.augmentations = augmentations
    
    def __call__(self, data: Data) -> Data:
        """
        Apply all augmentations sequentially.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph
        """
        for aug in self.augmentations:
            data = aug(data)
        return data


class RandomChoice:
    """
    Randomly choose one augmentation from a list.
    """
    
    def __init__(self, augmentations: list):
        """
        Initialize random choice.
        
        Args:
            augmentations: List of GraphAugmentation instances
        """
        self.augmentations = augmentations
    
    def __call__(self, data: Data) -> Data:
        """
        Apply one randomly chosen augmentation.
        
        Args:
            data: Input graph
            
        Returns:
            Augmented graph
        """
        aug = np.random.choice(self.augmentations)
        return aug(data)


class IdentityAugmentation(GraphAugmentation):
    """
    Identity augmentation (no change).
    Useful for baseline comparisons.
    """
    
    def __call__(self, data: Data) -> Data:
        """Return data unchanged."""
        return data


def get_augmentation(aug_type: str = "none", **kwargs) -> GraphAugmentation:
    """
    Factory function to get augmentation by name.
    
    Args:
        aug_type: Type of augmentation
        **kwargs: Arguments for augmentation
        
    Returns:
        GraphAugmentation instance
    """
    augmentations = {
        'none': IdentityAugmentation,
        'drop_nodes': DropNodes,
        'drop_edges': DropEdges,
        'mask_features': MaskNodeFeatures,
        'gaussian_noise': AddGaussianNoise,
        'attribute_noise': RandomAttributeNoise
    }
    
    if aug_type not in augmentations:
        raise ValueError(f"Unknown augmentation: {aug_type}. Choose from {list(augmentations.keys())}")
    
    return augmentations[aug_type](**kwargs)


def get_standard_augmentation(strength: str = "light") -> GraphAugmentation:
    """
    Get a standard augmentation pipeline.
    
    Args:
        strength: Augmentation strength ('light', 'medium', 'heavy')
        
    Returns:
        Composed augmentation
    """
    if strength == "light":
        return Compose([
            DropEdges(drop_prob=0.05),
            AddGaussianNoise(std=0.05)
        ])
    elif strength == "medium":
        return Compose([
            DropEdges(drop_prob=0.1),
            MaskNodeFeatures(mask_prob=0.1),
            AddGaussianNoise(std=0.1)
        ])
    elif strength == "heavy":
        return Compose([
            DropNodes(drop_prob=0.1),
            DropEdges(drop_prob=0.15),
            MaskNodeFeatures(mask_prob=0.15),
            AddGaussianNoise(std=0.15)
        ])
    else:
        raise ValueError(f"Unknown strength: {strength}")


if __name__ == "__main__":
    from .graph_builder import MoleculeGraphBuilder
    
    print("Testing graph augmentations...")
    
    # Create a sample graph
    builder = MoleculeGraphBuilder()
    smiles = "c1ccccc1"  # Benzene
    data = builder.smiles_to_graph(smiles)
    
    print(f"Original graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Original features shape: {data.x.shape}")
    
    # Test 1: Drop nodes
    print("\n1. Drop Nodes (10%)")
    aug = DropNodes(drop_prob=0.1)
    aug_data = aug(data)
    print(f"   After: {aug_data.num_nodes} nodes, {aug_data.num_edges} edges")
    
    # Test 2: Drop edges
    print("\n2. Drop Edges (20%)")
    aug = DropEdges(drop_prob=0.2)
    aug_data = aug(data)
    print(f"   After: {aug_data.num_nodes} nodes, {aug_data.num_edges} edges")
    
    # Test 3: Mask features
    print("\n3. Mask Node Features (15%)")
    aug = MaskNodeFeatures(mask_prob=0.15)
    aug_data = aug(data)
    masked_features = (aug_data.x == 0).sum().item()
    print(f"   Masked features: {masked_features}")
    
    # Test 4: Add Gaussian noise
    print("\n4. Add Gaussian Noise")
    aug = AddGaussianNoise(std=0.1)
    aug_data = aug(data)
    diff = torch.abs(aug_data.x - data.x).mean().item()
    print(f"   Average feature difference: {diff:.4f}")
    
    # Test 5: Compose augmentations
    print("\n5. Composed Augmentation")
    aug = Compose([
        DropEdges(drop_prob=0.1),
        AddGaussianNoise(std=0.05),
        MaskNodeFeatures(mask_prob=0.1)
    ])
    aug_data = aug(data)
    print(f"   After: {aug_data.num_nodes} nodes, {aug_data.num_edges} edges")
    
    # Test 6: Standard pipelines
    print("\n6. Standard Augmentation Pipelines")
    for strength in ["light", "medium", "heavy"]:
        aug = get_standard_augmentation(strength)
        aug_data = aug(data)
        print(f"   {strength.capitalize()}: {aug_data.num_edges} edges")
    
    print("\n Augmentation tests complete!")
