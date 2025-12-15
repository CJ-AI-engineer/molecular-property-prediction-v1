"""
Graph builder for converting SMILES strings to PyTorch Geometric Data objects.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing import Optional

from .featurizers import AtomFeaturizer, BondFeaturizer


class MoleculeGraphBuilder:
    """
    Convert SMILES strings to PyTorch Geometric Data objects.
    
    The graph is constructed as:
    - Nodes: Atoms with features from AtomFeaturizer
    - Edges: Bonds with features from BondFeaturizer (bidirectional)
    """
    
    def __init__(
        self,
        atom_featurizer: Optional[AtomFeaturizer] = None,
        bond_featurizer: Optional[BondFeaturizer] = None,
        explicit_hydrogens: bool = False
    ):
        """
        Initialize graph builder.
        
        Args:
            atom_featurizer: AtomFeaturizer instance. If None, uses default.
            bond_featurizer: BondFeaturizer instance. If None, uses default.
            explicit_hydrogens: Whether to add explicit hydrogens to molecule
        """
        self.atom_featurizer = atom_featurizer or AtomFeaturizer()
        self.bond_featurizer = bond_featurizer or BondFeaturizer()
        self.explicit_hydrogens = explicit_hydrogens
        
        self.node_feat_dim = self.atom_featurizer.get_feature_size()
        self.edge_feat_dim = self.bond_featurizer.get_feature_size()
    
    def smiles_to_graph(
        self,
        smiles: str,
        target: Optional[float] = None
    ) -> Data:
        """
        Convert a SMILES string to a PyTorch Geometric Data object.
        
        Args:
            smiles: SMILES string representation of molecule
            target: Optional target value (e.g., property to predict)
            
        Returns:
            PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_feat_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feat_dim]
                - y: Target value (if provided)
                - smiles: Original SMILES string
                
        Raises:
            ValueError: If SMILES is invalid
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Add explicit hydrogens if requested ... refer to page 69 - 71 of the lesson
        if self.explicit_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Extract node features
        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(self.atom_featurizer.featurize(atom))
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edge features
        # Create bidirectional edges (undirected graph)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            # Get atom indices
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Get bond features
            bond_feat = self.bond_featurizer.featurize(bond)
            
            # Add both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            edge_features.append(bond_feat)
            edge_features.append(bond_feat) 
        
        # Convert to tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # Handle single-atom molecules (no bonds)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.edge_feat_dim), dtype=torch.float)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )
        
        # Add target if provided
        if target is not None:
            data.y = torch.tensor([target], dtype=torch.float)
        
        return data
    
    def batch_smiles_to_graphs(
        self,
        smiles_list: list,
        targets: Optional[list] = None
    ) -> list:
        """
        Convert multiple SMILES strings to graphs.
        
        Args:
            smiles_list: List of SMILES strings
            targets: Optional list of target values
            
        Returns:
            List of PyTorch Geometric Data objects
        """
        graphs = []
        failed_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            try:
                target = targets[idx] if targets is not None else None
                graph = self.smiles_to_graph(smiles, target)
                graphs.append(graph)
            except (ValueError, Exception) as e:
                failed_indices.append((idx, smiles, str(e)))
                continue
        
        if failed_indices:
            print(f"Warning: Failed to convert {len(failed_indices)} molecules")
            for idx, smiles, error in failed_indices[:5]:  # Show first 5
                print(f"  - Index {idx}: {smiles[:50]}... ({error})")
        
        return graphs
    
    def get_feature_dims(self):
        """
        Get the dimensions of node and edge features.
        
        Returns:
            Tuple of (node_feat_dim, edge_feat_dim)
        """
        return self.node_feat_dim, self.edge_feat_dim



if __name__ == "__main__":
    print("Testing MoleculeGraphBuilder...")
    
    builder = MoleculeGraphBuilder()
    
    print(f"Node feature dim: {builder.node_feat_dim}")
    print(f"Edge feature dim: {builder.edge_feat_dim}")
    
    print("\n1. Testing single molecule (Ethanol: CCO)")
    smiles = "CCO"
    target = 0.5
    
    graph = builder.smiles_to_graph(smiles, target)
    
    print(f"   Nodes: {graph.num_nodes}")
    print(f"   Edges: {graph.num_edges}")
    print(f"   Node features shape: {graph.x.shape}")
    print(f"   Edge features shape: {graph.edge_attr.shape}")
    print(f"   Target: {graph.y}")
    
    print("\n2. Testing batch conversion")
    smiles_list = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O", "INVALID"]
    targets = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    graphs = builder.batch_smiles_to_graphs(smiles_list, targets)
    print(f"   Successfully converted: {len(graphs)}/5 molecules")
    
    print("\n3. Testing complex molecule (Aspirin)")
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    graph = builder.smiles_to_graph(aspirin)
    
    print(f"   Nodes: {graph.num_nodes}")
    print(f"   Edges: {graph.num_edges}")
    

    print("\n4. Testing single atom (Helium)")
    try:
        graph = builder.smiles_to_graph("[He]")
        print(f"   Nodes: {graph.num_nodes}")
        print(f"   Edges: {graph.num_edges}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n Graph builder tests complete!")
