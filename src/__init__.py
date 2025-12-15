"""
Molecular Property Prediction with Graph Neural Networks

A comprehensive package for predicting molecular properties using GNNs
with uncertainty quantification, explainability, and active learning.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.models import MolecularGCN, MolecularGIN, MolecularGAT
from src.data import MolecularDataset, MoleculeGraphBuilder
from src.training import MolecularPropertyTrainer

__all__ = [
    "MolecularGCN",
    "MolecularGIN",
    "MolecularGAT",
    "MolecularDataset",
    "MoleculeGraphBuilder",
    "MolecularPropertyTrainer",
]
