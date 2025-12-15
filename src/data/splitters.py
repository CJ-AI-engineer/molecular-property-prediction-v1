"""
Data splitting strategies for molecular datasets.
Includes random, scaffold, and stratified splitting.
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch


class DataSplitter:
    """Base class for data splitting strategies."""
    
    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset into train/valid/test sets.
        
        Args:
            dataset: Dataset to split
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed
            
        Returns:
            Tuple of (train_indices, valid_indices, test_indices)
        """
        raise NotImplementedError


class RandomSplitter(DataSplitter):
    """
    Random splitting strategy.
    Randomly assigns molecules to train/valid/test sets.
    """
    
    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Randomly split dataset.
        
        Args:
            dataset: Dataset to split
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed
            
        Returns:
            Tuple of (train_indices, valid_indices, test_indices)
        """
        np.random.seed(seed)
        
        n = len(dataset)
        indices = np.random.permutation(n)
        
        train_size = int(frac_train * n)
        valid_size = int(frac_valid * n)
        
        train_idx = indices[:train_size].tolist()
        valid_idx = indices[train_size:train_size + valid_size].tolist()
        test_idx = indices[train_size + valid_size:].tolist()
        
        return train_idx, valid_idx, test_idx


class ScaffoldSplitter(DataSplitter):
    """
    Scaffold-based splitting strategy.
    Groups molecules by Murcko scaffold and splits to minimize
    scaffold overlap between train/valid/test sets.
    
    This is important for drug discovery as it tests generalization
    to new chemical scaffolds.
    """
    
    def generate_scaffold(self, smiles: str, include_chirality: bool = False) -> str:
        """
        Generate Murcko scaffold for a molecule.
        
        Args:
            smiles: SMILES string
            include_chirality: Whether to include chirality in scaffold
            
        Returns:
            Scaffold SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=include_chirality
            )
            return scaffold
        except:
            return ""
    
    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset by scaffold.
        
        Args:
            dataset: Dataset to split (must have .smiles attribute or Data.smiles)
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed
            
        Returns:
            Tuple of (train_indices, valid_indices, test_indices)
        """
        np.random.seed(seed)
        
        # Group molecules by scaffold
        scaffold_to_indices = defaultdict(list)
        
        for idx in range(len(dataset)):
            data = dataset[idx]
            smiles = data.smiles if hasattr(data, 'smiles') else None
            
            if smiles is None:
                raise ValueError(f"Data at index {idx} has no SMILES attribute")
            
            scaffold = self.generate_scaffold(smiles)
            scaffold_to_indices[scaffold].append(idx)
        
        # Sort scaffolds by size (largest first)
        scaffolds = list(scaffold_to_indices.keys())
        scaffold_sizes = [len(scaffold_to_indices[s]) for s in scaffolds]
        
        # Sort by size descending
        sorted_indices = np.argsort(scaffold_sizes)[::-1]
        scaffolds = [scaffolds[i] for i in sorted_indices]
        
        # Distribute scaffolds to sets
        n = len(dataset)
        train_size = int(frac_train * n)
        valid_size = int(frac_valid * n)
        
        train_idx, valid_idx, test_idx = [], [], []
        train_count, valid_count = 0, 0
        
        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]
            
            if train_count + len(indices) <= train_size:
                train_idx.extend(indices)
                train_count += len(indices)
            elif valid_count + len(indices) <= valid_size:
                valid_idx.extend(indices)
                valid_count += len(indices)
            else:
                test_idx.extend(indices)
        
        # Shuffle each split
        np.random.shuffle(train_idx)
        np.random.shuffle(valid_idx)
        np.random.shuffle(test_idx)
        
        return train_idx, valid_idx, test_idx


class StratifiedSplitter(DataSplitter):
    """
    Stratified splitting for classification tasks.
    Ensures class balance across train/valid/test sets.
    """
    
    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset with stratification by target class.
        
        Args:
            dataset: Dataset to split
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set  
            frac_test: Fraction for test set
            seed: Random seed
            
        Returns:
            Tuple of (train_indices, valid_indices, test_indices)
        """
        targets = []
        for idx in range(len(dataset)):
            data = dataset[idx]
            target = data.y.item() if hasattr(data, 'y') else 0
            targets.append(int(target))
        
        targets = np.array(targets)
        indices = np.arange(len(dataset))
        

        train_valid_idx, test_idx = train_test_split(
            indices,
            test_size=frac_test,
            stratify=targets,
            random_state=seed
        )
        

        train_targets = targets[train_valid_idx]
        valid_size = frac_valid / (frac_train + frac_valid)
        
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_size,
            stratify=train_targets,
            random_state=seed
        )
        
        return train_idx.tolist(), valid_idx.tolist(), test_idx.tolist()


class TemporalSplitter(DataSplitter):
    """
    Temporal splitting strategy.
    Splits by time/date to simulate real-world scenario where
    model is trained on older data and tested on newer data.
    """
    
    def split(
        self,
        dataset,
        frac_train: float = 0.8,
        frac_valid: float = 0.1,
        frac_test: float = 0.1,
        seed: int = 42,
        time_key: str = 'date'
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset temporally.
        
        Args:
            dataset: Dataset to split (must have time/date attribute)
            frac_train: Fraction for training set
            frac_valid: Fraction for validation set
            frac_test: Fraction for test set
            seed: Random seed (not used, kept for API consistency)
            time_key: Key to access time/date in data objects
            
        Returns:
            Tuple of (train_indices, valid_indices, test_indices)
        """
        # Extract timestamps
        times = []
        for idx in range(len(dataset)):
            data = dataset[idx]
            time = getattr(data, time_key, idx)  # Use index if no time
            times.append((idx, time))
        
        # Sort by time
        times.sort(key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in times]
        
        # Split chronologically
        n = len(dataset)
        train_size = int(frac_train * n)
        valid_size = int(frac_valid * n)
        
        train_idx = sorted_indices[:train_size]
        valid_idx = sorted_indices[train_size:train_size + valid_size]
        test_idx = sorted_indices[train_size + valid_size:]
        
        return train_idx, valid_idx, test_idx


def get_splitter(strategy: str = "random") -> DataSplitter:
    """
    Factory function to get splitter by name.
    
    Args:
        strategy: Splitting strategy ('random', 'scaffold', 'stratified', 'temporal')
        
    Returns:
        DataSplitter instance
    """
    splitters = {
        'random': RandomSplitter,
        'scaffold': ScaffoldSplitter,
        'stratified': StratifiedSplitter,
        'temporal': TemporalSplitter
    }
    
    if strategy not in splitters:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(splitters.keys())}")
    
    return splitters[strategy]()


if __name__ == "__main__":
    from .graph_builder import MoleculeGraphBuilder
    from .loaders import SimpleDataset
    
    print("Testing data splitters...")
    
    builder = MoleculeGraphBuilder()
    smiles_list = [
        "CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "CCN",
        "c1cccnc1", "CC(C)C", "CCCC", "c1ccco1", "CCCN"
    ]
    targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Binary classification
    
    graphs = builder.batch_smiles_to_graphs(smiles_list, targets)
    dataset = SimpleDataset(graphs)
    
    print(f"Dataset size: {len(dataset)}\n")
    
    # Test 1: Random splitting
    print("1. Random Splitter")
    splitter = RandomSplitter()
    train, valid, test = splitter.split(dataset)
    print(f"   Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    
    # Test 2: Scaffold splitting
    print("\n2. Scaffold Splitter")
    splitter = ScaffoldSplitter()
    train, valid, test = splitter.split(dataset)
    print(f"   Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    
    # Test 3: Stratified splitting
    print("\n3. Stratified Splitter")
    splitter = StratifiedSplitter()
    train, valid, test = splitter.split(dataset)
    print(f"   Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    
    # Check class balance
    train_targets = [dataset[i].y.item() for i in train]
    print(f"   Train class balance: {sum(train_targets)}/{len(train_targets)}")
    
    # Test 4: Factory function
    print("\n4. Using factory function")
    splitter = get_splitter("random")
    train, valid, test = splitter.split(dataset)
    print(f"   Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")
    
    print("\n Splitter tests complete!")
