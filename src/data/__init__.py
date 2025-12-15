"""
Data processing module for molecular property prediction.

This module provides tools for:
- Featurizing molecules (atoms and bonds)
- Converting SMILES to graphs
- Loading and splitting datasets
- Data augmentation
- Classical molecular descriptors
"""

from .featurizers import AtomFeaturizer, BondFeaturizer
from .graph_builder import MoleculeGraphBuilder
from .loaders import (
    MolecularDataset,
    SimpleDataset,
    CSVMolecularDataset,
    get_dataloader
)
from .splitters import (
    RandomSplitter,
    ScaffoldSplitter,
    StratifiedSplitter,
    TemporalSplitter,
    get_splitter
)
from .augmentation import (
    DropNodes,
    DropEdges,
    MaskNodeFeatures,
    AddGaussianNoise,
    Compose,
    RandomChoice,
    get_augmentation,
    get_standard_augmentation
)
from .molecular_descriptors import (
    MorganFingerprintGenerator,
    MACCSKeysGenerator,
    RDKitFingerprintGenerator,
    PhysicochemicalDescriptors,
    CombinedDescriptors,
    get_descriptor_generator
)

__all__ = [
    # Featurizers
    'AtomFeaturizer',
    'BondFeaturizer',
    
    # Graph Builder
    'MoleculeGraphBuilder',
    
    # Loaders
    'MolecularDataset',
    'SimpleDataset',
    'CSVMolecularDataset',
    'get_dataloader',
    
    # Splitters
    'RandomSplitter',
    'ScaffoldSplitter',
    'StratifiedSplitter',
    'TemporalSplitter',
    'get_splitter',
    
    # Augmentation
    'DropNodes',
    'DropEdges',
    'MaskNodeFeatures',
    'AddGaussianNoise',
    'Compose',
    'RandomChoice',
    'get_augmentation',
    'get_standard_augmentation',
    
    # Descriptors
    'MorganFingerprintGenerator',
    'MACCSKeysGenerator',
    'RDKitFingerprintGenerator',
    'PhysicochemicalDescriptors',
    'CombinedDescriptors',
    'get_descriptor_generator',
]
