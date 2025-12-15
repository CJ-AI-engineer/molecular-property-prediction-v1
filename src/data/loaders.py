"""
Dataset loaders for molecular property prediction.
"""

import os
import pickle
from pathlib import Path
from typing import Optional, List, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .graph_builder import MoleculeGraphBuilder
from .featurizers import AtomFeaturizer, BondFeaturizer


class MolecularDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for molecular property prediction.
    
    Loads preprocessed graph data or processes raw SMILES on the fly.
    """
    
    def __init__(
        self,
        root: str,
        dataset_name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Initialize molecular dataset.
        
        Args:
            root: Root directory where the dataset is stored
            dataset_name: Name of the dataset (BBBP, HIV, ESOL, FreeSolv)
            transform: Optional transform to apply on-the-fly
            pre_transform: Optional transform to apply before saving
            pre_filter: Optional filter to apply before saving
        """
        self.dataset_name = dataset_name
        self.root = root
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to check for."""
        return [f"{self.dataset_name}_processed.pt"]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files to check for."""
        return [f"{self.dataset_name}_processed.pt"]
    
    def download(self):
        """Download raw data (if needed)."""
        # Data should be preprocessed by preprocess_data.py
        pass
    
    def process(self):
        """Process raw data into graphs."""
        # Check if preprocessed file exists
        raw_path = os.path.join(self.root, f"{self.dataset_name}_processed.pt")
        
        if os.path.exists(raw_path):
            data_list = torch.load(raw_path, weights_only=False)
        else:
            raise FileNotFoundError(
                f"Preprocessed file not found: {raw_path}\n"
                f"Please run: python scripts/preprocess_data.py --dataset {self.dataset_name}"
            )
        
        # Apply pre_filter
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        # Apply pre_transform
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    @property
    def num_node_features(self) -> int:
        """Number of node features."""
        return self.data.x.shape[1] if self.data.x is not None else 0
    
    @property
    def num_edge_features(self) -> int:
        """Number of edge features."""
        return self.data.edge_attr.shape[1] if self.data.edge_attr is not None else 0
    
    @property
    def num_tasks(self) -> int:
        """Number of prediction tasks."""
        return self.data.y.shape[1] if len(self.data.y.shape) > 1 else 1


class SimpleDataset(Dataset):
    """
    Simple PyTorch Dataset for molecular graphs.
    Useful when you have a list of Data objects.
    """
    
    def __init__(
        self,
        data_list: List[Data],
        transform: Optional[Callable] = None
    ):
        """
        Initialize simple dataset.
        
        Args:
            data_list: List of PyTorch Geometric Data objects
            transform: Optional transform to apply
        """
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data


class CSVMolecularDataset(Dataset):
    """
    Dataset that loads SMILES from CSV and converts to graphs on-the-fly.
    Useful for smaller datasets or when you want to experiment with featurization.
    """
    
    def __init__(
        self,
        csv_path: str,
        smiles_col: str = "smiles",
        target_col: Optional[str] = None,
        graph_builder: Optional[MoleculeGraphBuilder] = None,
        transform: Optional[Callable] = None,
        cache_graphs: bool = True
    ):
        """
        Initialize CSV-based dataset.
        
        Args:
            csv_path: Path to CSV file
            smiles_col: Name of column containing SMILES strings
            target_col: Name of column containing target values (optional)
            graph_builder: MoleculeGraphBuilder instance
            transform: Optional transform to apply
            cache_graphs: Whether to cache converted graphs in memory
        """
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.transform = transform
        self.cache_graphs = cache_graphs
        
        self.df = pd.read_csv(csv_path)
        
        self.graph_builder = graph_builder or MoleculeGraphBuilder()
        
        self._graph_cache = {} if cache_graphs else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self._graph_cache is not None and idx in self._graph_cache:
            data = self._graph_cache[idx]
        else:
            # Get SMILES and target
            row = self.df.iloc[idx]
            smiles = row[self.smiles_col]
            target = row[self.target_col] if self.target_col else None
            
            try:
                data = self.graph_builder.smiles_to_graph(smiles, target)
                data.idx = idx  
                
                if self._graph_cache is not None:
                    self._graph_cache[idx] = data
                    
            except Exception as e:
                raise ValueError(f"Failed to convert molecule at index {idx}: {e}")
        

        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    @property
    def num_node_features(self):
        """Get node feature dimension from first molecule."""
        return self[0].num_node_features
    
    @property
    def num_edge_features(self):
        """Get edge feature dimension from first molecule."""
        return self[0].num_edge_features if self[0].edge_attr is not None else 0


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a PyTorch Geometric DataLoader.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        PyTorch Geometric DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


if __name__ == "__main__":
    import tempfile
    
    print("Testing MolecularDataset loaders...")
    
    print("\n1. Testing SimpleDataset")
    from .graph_builder import MoleculeGraphBuilder
    
    builder = MoleculeGraphBuilder()
    

    smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]
    targets = [0.1, 0.2, 0.3]
    graphs = builder.batch_smiles_to_graphs(smiles_list, targets)
    
    dataset = SimpleDataset(graphs)
    print(f"   Dataset size: {len(dataset)}")
    print(f"   First item nodes: {dataset[0].num_nodes}")
    
    loader = get_dataloader(dataset, batch_size=2)
    batch = next(iter(loader))
    print(f"   Batch size: {batch.num_graphs}")
    print(f"   Batch nodes: {batch.num_nodes}")
    
    print("\n2. Testing CSVMolecularDataset")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("smiles,target\n")
        f.write("CCO,0.1\n")
        f.write("CC(C)O,0.2\n")
        f.write("c1ccccc1,0.3\n")
        temp_csv = f.name
    
    try:
        dataset = CSVMolecularDataset(
            temp_csv,
            smiles_col="smiles",
            target_col="target",
            cache_graphs=True
        )
        
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Node features: {dataset.num_node_features}")
        print(f"   Edge features: {dataset.num_edge_features}")
        
        # Test caching
        data1 = dataset[0]
        data2 = dataset[0]
        print(f"   Caching works: {data1 is data2}")
        
    finally:
        os.unlink(temp_csv)
    
    print("\n Loader tests complete!")

