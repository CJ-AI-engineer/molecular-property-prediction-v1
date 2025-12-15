"""
Classical molecular descriptors and fingerprints.
Used as features for baseline machine learning models (RF, SVM, etc.)
"""

import numpy as np
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, MolSurf
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.ML.Descriptors import MoleculeDescriptors


class MorganFingerprintGenerator:
    """
    Generate Morgan (ECFP) fingerprints.
    Most popular fingerprint for molecular ML.
    """
    
    def __init__(self, radius: int = 2, n_bits: int = 2048, use_features: bool = False):
        """
        Initialize Morgan fingerprint generator.
        
        Args:
            radius: Radius of the fingerprint (typically 2 or 3)
            n_bits: Number of bits in the fingerprint
            use_features: Use feature-based fingerprints instead of connectivity
        """
        self.radius = radius
        self.n_bits = n_bits
        self.use_features = use_features
    
    def transform(self, smiles: str) -> np.ndarray:
        """
        Generate Morgan fingerprint for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Binary fingerprint as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.n_bits,
            useFeatures=self.use_features
        )
        
        return np.array(fp)
    
    def batch_transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate fingerprints for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Array of shape (n_molecules, n_bits)
        """
        return np.array([self.transform(s) for s in smiles_list])


class MACCSKeysGenerator:
    """
    Generate MACCS keys fingerprints.
    166-bit structural key descriptors.
    """
    
    def __init__(self):
        """Initialize MACCS keys generator."""
        self.n_bits = 166
    
    def transform(self, smiles: str) -> np.ndarray:
        """
        Generate MACCS keys for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Binary fingerprint as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits)
        
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return np.array(fp)
    
    def batch_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Generate fingerprints for multiple molecules."""
        return np.array([self.transform(s) for s in smiles_list])


class RDKitFingerprintGenerator:
    """
    Generate RDKit topological fingerprints.
    Path-based fingerprints.
    """
    
    def __init__(self, n_bits: int = 2048, max_path: int = 7):
        """
        Initialize RDKit fingerprint generator.
        
        Args:
            n_bits: Number of bits in the fingerprint
            max_path: Maximum path length
        """
        self.n_bits = n_bits
        self.max_path = max_path
    
    def transform(self, smiles: str) -> np.ndarray:
        """
        Generate RDKit fingerprint for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Binary fingerprint as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits)
        
        fp = FingerprintMols.FingerprintMol(
            mol,
            fpSize=self.n_bits,
            maxPath=self.max_path
        )
        
        return np.array(fp)
    
    def batch_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Generate fingerprints for multiple molecules."""
        return np.array([self.transform(s) for s in smiles_list])


class PhysicochemicalDescriptors:
    """
    Generate physicochemical descriptors.
    Interpretable molecular properties.
    """
    
    def __init__(self, descriptor_names: Optional[List[str]] = None):
        """
        Initialize descriptor generator.
        
        Args:
            descriptor_names: List of descriptor names to compute.
                            If None, uses a default set of important descriptors.
        """
        if descriptor_names is None:
            # Default set of important descriptors
            self.descriptor_names = [
                'MolWt',           # Molecular weight
                'MolLogP',         # Partition coefficient
                'NumHDonors',      # Hydrogen bond donors
                'NumHAcceptors',   # Hydrogen bond acceptors
                'NumRotatableBonds', # Rotatable bonds
                'TPSA',            # Topological polar surface area
                'NumAromaticRings', # Aromatic rings
                'NumAliphaticRings', # Aliphatic rings
                'FractionCsp3',    # Fraction of sp3 carbons
                'RingCount',       # Number of rings
            ]
        else:
            self.descriptor_names = descriptor_names
        
        self.n_features = len(self.descriptor_names)
    
    def _compute_descriptors(self, mol) -> np.ndarray:
        """Compute descriptors for a molecule."""
        descriptors = []
        
        for name in self.descriptor_names:
            try:
                if name == 'MolWt':
                    val = Descriptors.MolWt(mol)
                elif name == 'MolLogP':
                    val = Descriptors.MolLogP(mol)
                elif name == 'NumHDonors':
                    val = Descriptors.NumHDonors(mol)
                elif name == 'NumHAcceptors':
                    val = Descriptors.NumHAcceptors(mol)
                elif name == 'NumRotatableBonds':
                    val = Descriptors.NumRotatableBonds(mol)
                elif name == 'TPSA':
                    val = Descriptors.TPSA(mol)
                elif name == 'NumAromaticRings':
                    val = Descriptors.NumAromaticRings(mol)
                elif name == 'NumAliphaticRings':
                    val = Descriptors.NumAliphaticRings(mol)
                elif name == 'FractionCsp3':
                    val = Descriptors.FractionCSP3(mol)
                elif name == 'RingCount':
                    val = Descriptors.RingCount(mol)
                else:
                    # Try to get from Descriptors module
                    val = getattr(Descriptors, name)(mol)
                
                descriptors.append(float(val))
            except:
                descriptors.append(0.0)
        
        return np.array(descriptors)
    
    def transform(self, smiles: str) -> np.ndarray:
        """
        Compute physicochemical descriptors for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Descriptor vector as numpy array
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_features)
        
        return self._compute_descriptors(mol)
    
    def batch_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Compute descriptors for multiple molecules."""
        return np.array([self.transform(s) for s in smiles_list])


class CombinedDescriptors:
    """
    Combine multiple descriptor types.
    Useful for comprehensive feature representation.
    """
    
    def __init__(
        self,
        use_morgan: bool = True,
        use_maccs: bool = False,
        use_rdkit: bool = False,
        use_physchem: bool = True,
        morgan_radius: int = 2,
        morgan_bits: int = 2048
    ):
        """
        Initialize combined descriptor generator.
        
        Args:
            use_morgan: Include Morgan fingerprints
            use_maccs: Include MACCS keys
            use_rdkit: Include RDKit fingerprints
            use_physchem: Include physicochemical descriptors
            morgan_radius: Radius for Morgan fingerprints
            morgan_bits: Number of bits for Morgan fingerprints
        """
        self.generators = []
        
        if use_morgan:
            self.generators.append(
                ('morgan', MorganFingerprintGenerator(morgan_radius, morgan_bits))
            )
        
        if use_maccs:
            self.generators.append(
                ('maccs', MACCSKeysGenerator())
            )
        
        if use_rdkit:
            self.generators.append(
                ('rdkit', RDKitFingerprintGenerator())
            )
        
        if use_physchem:
            self.generators.append(
                ('physchem', PhysicochemicalDescriptors())
            )
        
        # Calculate total feature dimension
        self.n_features = sum(
            gen.n_bits if hasattr(gen, 'n_bits') else gen.n_features
            for _, gen in self.generators
        )
    
    def transform(self, smiles: str) -> np.ndarray:
        """
        Generate combined descriptors for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Combined descriptor vector
        """
        features = []
        
        for name, gen in self.generators:
            feat = gen.transform(smiles)
            features.append(feat)
        
        return np.concatenate(features)
    
    def batch_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Generate descriptors for multiple molecules."""
        return np.array([self.transform(s) for s in smiles_list])


def get_descriptor_generator(
    descriptor_type: str = "morgan",
    **kwargs
):
    """
    Factory function to get descriptor generator by name.
    
    Args:
        descriptor_type: Type of descriptor
        **kwargs: Arguments for the generator
        
    Returns:
        Descriptor generator instance
    """
    generators = {
        'morgan': MorganFingerprintGenerator,
        'maccs': MACCSKeysGenerator,
        'rdkit': RDKitFingerprintGenerator,
        'physchem': PhysicochemicalDescriptors,
        'combined': CombinedDescriptors
    }
    
    if descriptor_type not in generators:
        raise ValueError(
            f"Unknown descriptor type: {descriptor_type}. "
            f"Choose from {list(generators.keys())}"
        )
    
    return generators[descriptor_type](**kwargs)


if __name__ == "__main__":
    print("Testing molecular descriptors...")
    
    # Sample molecules
    smiles_list = [
        "CCO",           # Ethanol
        "c1ccccc1",      # Benzene
        "CC(=O)O",       # Acetic acid
        "CC(C)O"         # Isopropanol
    ]
    
    # Test 1: Morgan fingerprints
    print("\n1. Morgan Fingerprints")
    morgan = MorganFingerprintGenerator(radius=2, n_bits=2048)
    fps = morgan.batch_transform(smiles_list)
    print(f"   Shape: {fps.shape}")
    print(f"   Sparsity: {(fps == 0).sum() / fps.size * 100:.1f}% zeros")
    
    # Test 2: MACCS keys
    print("\n2. MACCS Keys")
    maccs = MACCSKeysGenerator()
    fps = maccs.batch_transform(smiles_list)
    print(f"   Shape: {fps.shape}")
    
    # Test 3: Physicochemical descriptors
    print("\n3. Physicochemical Descriptors")
    physchem = PhysicochemicalDescriptors()
    descs = physchem.batch_transform(smiles_list)
    print(f"   Shape: {descs.shape}")
    print(f"   Descriptor names: {physchem.descriptor_names[:5]}...")
    
    # Show values for ethanol
    print(f"\n   Ethanol descriptors:")
    for name, val in zip(physchem.descriptor_names, descs[0]):
        print(f"     {name}: {val:.2f}")
    
    # Test 4: Combined descriptors
    print("\n4. Combined Descriptors")
    combined = CombinedDescriptors(
        use_morgan=True,
        use_physchem=True
    )
    features = combined.batch_transform(smiles_list)
    print(f"   Shape: {features.shape}")
    print(f"   Total features: {combined.n_features}")
    
    # Test 5: Factory function
    print("\n5. Using Factory Function")
    gen = get_descriptor_generator("morgan", radius=3, n_bits=1024)
    fps = gen.batch_transform(smiles_list)
    print(f"   Shape: {fps.shape}")
    
    print("\n Descriptor tests complete!")
