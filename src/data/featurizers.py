"""
Molecular featurizers for converting atoms and bonds to numerical features.
Used by graph_builder to create node and edge features for GNNs.
"""

from typing import List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class AtomFeaturizer: # Ask Madam if I should add more features here 
    """
    Extract atom-level features for graph neural networks.
    
    Features extracted:
    - Atomic number (one-hot or integer)
    - Degree (number of bonds)
    - Formal charge
    - Hybridization (sp, sp2, sp3, etc.)
    - Aromaticity
    - Number of hydrogens
    - Chirality
    - Radical electrons
    - Is in ring
    """
    
    def __init__(
        self,
        allowable_atoms: Optional[List[str]] = None,
        use_chirality: bool = True,
        use_partial_charge: bool = False
    ):
        """
        Initialize atom featurizer.
        
        Args:
            allowable_atoms: List of atom symbols to one-hot encode.
                           If None, uses common organic atoms.
            use_chirality: Whether to include chirality features
            use_partial_charge: Whether to compute partial charges (slower)
        """
        if allowable_atoms is None:
            # I am using only Common atoms in drug-like molecules
            self.allowable_atoms = [
                'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I',
                'B', 'Si', 'Se', 'H', 'Unknown'
            ]
        else:
            self.allowable_atoms = allowable_atoms + ['Unknown']
        
        self.use_chirality = use_chirality
        self.use_partial_charge = use_partial_charge
        
        self.feature_size = self._calculate_feature_size()
    
    def _calculate_feature_size(self) -> int:  # Check the details from the lesson 5 topic 3
        """Calculate total feature vector size."""
        size = 0
        size += len(self.allowable_atoms)  # Atom type (one-hot)
        size += 7  # Degree (0-6)
        size += 5  # Formal charge (-2 to +2)
        size += 6  # Hybridization (SP, SP2, SP3, SP3D, SP3D2, other)
        size += 1  # Aromaticity
        size += 5  # Number of hydrogens (0-4)
        size += 1  # Is in ring
        size += 1  # Radical electrons
        
        if self.use_chirality:
            size += 4  # Chirality (R, S, unspecified, other)
        
        if self.use_partial_charge:
            size += 1  # Partial charge (continuous)
        
        return size
    
    def _one_hot_encode(self, value, allowable_set):
        """One-hot encode a value given an allowable set."""
        encoding = [0] * len(allowable_set)
        if value not in allowable_set:
            value = allowable_set[-1]  # this is the Unknown category
        encoding[allowable_set.index(value)] = 1
        return encoding
    
    def featurize(self, atom: Chem.Atom) -> np.ndarray:
        """
        Convert an RDKit Atom to a feature vector.
        
        Args:
            atom: RDKit Atom object
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Atom type (one-hot)
        symbol = atom.GetSymbol()
        if symbol not in self.allowable_atoms[:-1]:
            symbol = 'Unknown'
        features.extend(self._one_hot_encode(symbol, self.allowable_atoms))
        
        # Degree (number of bonds)
        degree = atom.GetDegree()
        features.extend(self._one_hot_encode(
            min(degree, 6), [0, 1, 2, 3, 4, 5, 6]
        ))
        
        # Formal charge
        formal_charge = atom.GetFormalCharge()
        features.extend(self._one_hot_encode(
            formal_charge, [-2, -1, 0, 1, 2]
        ))
        
        # Hybridization
        hybridization = str(atom.GetHybridization())
        features.extend(self._one_hot_encode(
            hybridization,
            ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'other']
        ))
        
        # Aromaticity
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # Number of hydrogens
        num_hs = atom.GetTotalNumHs()
        features.extend(self._one_hot_encode(
            min(num_hs, 4), [0, 1, 2, 3, 4]
        ))
        
        # Is in ring
        features.append(1 if atom.IsInRing() else 0)
        
        # Radical electrons
        features.append(atom.GetNumRadicalElectrons())
        
        # Chirality
        if self.use_chirality:
            try:
                chirality = str(atom.GetChiralTag())
                features.extend(self._one_hot_encode(
                    chirality,
                    ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                     'CHI_TETRAHEDRAL_CCW', 'other']
                ))
            except:
                features.extend([1, 0, 0, 0])  # Unspecified
        
        # Partial charge (requires Gasteiger charges) ... confirm form the lesson text (important)
        if self.use_partial_charge:
            try:
                mol = atom.GetOwningMol()
                AllChem.ComputeGasteigerCharges(mol)
                charge = float(atom.GetProp('_GasteigerCharge'))
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
                features.append(charge)
            except:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_size(self) -> int:
        """Return the size of the feature vector."""
        return self.feature_size


class BondFeaturizer:
    """
    Extract bond-level features for graph neural networks.
    
    Features extracted:
    - Bond type (single, double, triple, aromatic)
    - Conjugation
    - Is in ring
    - Stereochemistry (E/Z)
    """
    
    def __init__(self, use_stereochemistry: bool = True):
        """
        Initialize bond featurizer.
        
        Args:
            use_stereochemistry: Whether to include stereochemistry features
        """
        self.use_stereochemistry = use_stereochemistry
        self.feature_size = self._calculate_feature_size()
    
    def _calculate_feature_size(self) -> int:
        """Calculate total feature vector size."""
        size = 0
        size += 5  # Bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC, other)
        size += 1  # Conjugation
        size += 1  # Is in ring
        
        if self.use_stereochemistry:
            size += 6  # Stereo (STEREONONE, STEREOANY, STEREOZ, STEREOE, etc.)
        
        return size
    
    def _one_hot_encode(self, value, allowable_set):
        """One-hot encode a value given an allowable set."""
        encoding = [0] * len(allowable_set)
        if value not in allowable_set:
            value = allowable_set[-1]  # Unknown/other category
        encoding[allowable_set.index(value)] = 1
        return encoding
    
    def featurize(self, bond: Chem.Bond) -> np.ndarray:
        """
        Convert an RDKit Bond to a feature vector.
        
        Args:
            bond: RDKit Bond object
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Bond type
        bond_type = str(bond.GetBondType())
        features.extend(self._one_hot_encode(
            bond_type,
            ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'other']
        ))
        
        # Conjugation
        features.append(1 if bond.GetIsConjugated() else 0)
        
        # Is in ring
        features.append(1 if bond.IsInRing() else 0)
        
        # Stereochemistry
        if self.use_stereochemistry:
            stereo = str(bond.GetStereo())
            features.extend(self._one_hot_encode(
                stereo,
                ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE',
                 'STEREOCIS', 'STEREOTRANS']
            ))
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_size(self) -> int:
        """Return the size of the feature vector."""
        return self.feature_size
