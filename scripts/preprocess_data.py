"""
Preprocess molecular datasets: SMILES → Graph objects
Saves processed data in PyTorch Geometric format
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

# Add src to path otherwise problem in running
sys.path.append(str(Path(__file__).parent.parent))

from src.data.featurizers import AtomFeaturizer, BondFeaturizer
from src.data.graph_builder import MoleculeGraphBuilder
from torch_geometric.data import Data, InMemoryDataset


def load_dataset(dataset_name, data_dir="data/raw"):
    """Load raw dataset CSV."""
    
    dataset_paths = {
        'BBBP': f"{data_dir}/BBBP/BBBP.csv",
        'HIV': f"{data_dir}/HIV/HIV.csv",
        'ESOL': f"{data_dir}/ESOL/ESOL.csv",
        'FreeSolv': f"{data_dir}/FreeSolv/FreeSolv.csv",
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    path = dataset_paths[dataset_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\nRun scripts/download_data.sh first")
    
    df = pd.read_csv(path)
    print(f"Loaded {dataset_name}: {len(df)} molecules")
    
    return df


def preprocess_dataset(dataset_name, output_dir="data/processed"):
    """
    Convert SMILES to graph objects and save.
    """
    
    print(f"\n{'='*60}")
    print(f"Preprocessing {dataset_name} Dataset")
    print(f"{'='*60}\n")
    
    df = load_dataset(dataset_name)
    
    # Initialize featurizers and graph builder
    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()
    graph_builder = MoleculeGraphBuilder(atom_featurizer, bond_featurizer)
    
    # Column names differ by dataset
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    
    # Target column (dataset-specific)
    target_cols = {
        'BBBP': 'p_np',
        'HIV': 'HIV_active',
        'ESOL': 'measured log solubility in mols per litre',
        'FreeSolv': 'expt',
    }
    target_col = target_cols.get(dataset_name, df.columns[-1])
    
    # Process molecules
    data_list = []
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting SMILES"):
        smiles = row[smiles_col]
        target = row[target_col]
        
        try:
            # Build graph
            data = graph_builder.smiles_to_graph(smiles)
            
            # Add target
            data.y = torch.tensor([target], dtype=torch.float)
            
            # Add metadata - Ask Michelle Madam on this for detals
            data.smiles = smiles
            data.idx = idx
            
            data_list.append(data)
            
        except Exception as e:
            failed.append((idx, smiles, str(e)))
            continue
    
    print(f"\n Successfully processed: {len(data_list)} molecules")
    print(f"✗ Failed: {len(failed)} molecules")
    
    if failed and len(failed) < 10:
        print("\nFailed molecules:")
        for idx, smiles, error in failed[:5]:
            print(f"  - {idx}: {smiles[:50]}... ({error})")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_processed.pt")
    
    torch.save(data_list, output_path)
    print(f"\n Saved to: {output_path}")
    

    print("\nDataset Statistics:")
    print(f"  - Number of graphs: {len(data_list)}")
    print(f"  - Avg nodes per graph: {sum(d.num_nodes for d in data_list) / len(data_list):.1f}")
    print(f"  - Avg edges per graph: {sum(d.num_edges for d in data_list) / len(data_list):.1f}")
    print(f"  - Node feature dim: {data_list[0].x.shape[1]}")
    print(f"  - Edge feature dim: {data_list[0].edge_attr.shape[1] if data_list[0].edge_attr is not None else 0}")
    
    targets = torch.cat([d.y for d in data_list])
    print(f"\nTarget Statistics:")
    
    if dataset_name in ['BBBP', 'HIV']:  # Classification
        print(f"  - Class 0: {(targets == 0).sum().item()} ({(targets == 0).float().mean() * 100:.1f}%)")
        print(f"  - Class 1: {(targets == 1).sum().item()} ({(targets == 1).float().mean() * 100:.1f}%)")
    else:  
        print(f"  - Mean: {targets.mean().item():.3f}")
        print(f"  - Std: {targets.std().item():.3f}")
        print(f"  - Min: {targets.min().item():.3f}")
        print(f"  - Max: {targets.max().item():.3f}")
    
    return data_list


def main():
    parser = argparse.ArgumentParser(description="Preprocess molecular datasets")
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['BBBP', 'HIV', 'ESOL', 'FreeSolv', 'all'],
        default='all',
        help='Dataset to preprocess'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    args = parser.parse_args()
    
    datasets = ['BBBP', 'HIV', 'ESOL', 'FreeSolv'] if args.dataset == 'all' else [args.dataset]
    
    print("\n" + "="*60)
    print("MOLECULAR DATASET PREPROCESSING")
    print("="*60)
    
    for dataset in datasets:
        try:
            preprocess_dataset(dataset, args.output_dir)
        except Exception as e:
            print(f"\n✗ Error processing {dataset}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python scripts/train.py --dataset BBBP --model gcn")
    print("2. Or open notebooks/01_data_exploration.ipynb")
    print()


if __name__ == "__main__":
    main()
