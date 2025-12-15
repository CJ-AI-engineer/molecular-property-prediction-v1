"""
Reproducibility utilities.
Set random seeds for PyTorch, NumPy, and Python random.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and GPU)
    - PyTorch backends (cuDNN)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_random_state():
    """
    Get current random state for all RNGs.
    
    Returns:
        Dictionary with random states
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict):
    """
    Restore random state for all RNGs.
    
    Args:
        state: Dictionary with random states (from get_random_state)
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def make_deterministic(seed: int = 42, warn: bool = True):
    """
    Make training fully deterministic.
    
    Note: This may reduce performance but ensures reproducibility.
    
    Args:
        seed: Random seed
        warn: Whether to print warning about performance
    """
    set_seed(seed)
    
    torch.use_deterministic_algorithms(True)
    
    if warn:
        print("Warning: Deterministic mode enabled. This may reduce performance.")
        print("Some operations may not be supported in deterministic mode.")


if __name__ == "__main__":
    print("Testing seed utilities...")
    
    print("\n1. Testing set_seed()")
    set_seed(42)
    
    python_rand = random.random()
    numpy_rand = np.random.rand()
    torch_rand = torch.rand(1).item()
    
    print(f"   Python random: {python_rand:.6f}")
    print(f"   NumPy random: {numpy_rand:.6f}")
    print(f"   PyTorch random: {torch_rand:.6f}")
    
    print("\n2. Testing Reproducibility")
    
    set_seed(42)
    nums1 = [random.random() for _ in range(3)]
    
    set_seed(42)
    nums2 = [random.random() for _ in range(3)]
    
    print(f"   First run: {nums1}")
    print(f"   Second run: {nums2}")
    print(f"   Same results: {nums1 == nums2}")
    
    print("\n3. Testing Random State Save/Restore")
    
    set_seed(42)
    before = [random.random() for _ in range(2)]
    
    state = get_random_state()
    
    middle = [random.random() for _ in range(2)]
    
    set_random_state(state)
    
    after = [random.random() for _ in range(2)]
    
    print(f"   Before state save: {before}")
    print(f"   After state save: {middle}")
    print(f"   After state restore: {after}")
    print(f"   Restored correctly: {middle == after}")
    
    print("\n4. Testing PyTorch Operations")
    
    set_seed(42)
    tensor1 = torch.randn(3, 3)
    
    set_seed(42)
    tensor2 = torch.randn(3, 3)
    
    print(f"   Tensors equal: {torch.allclose(tensor1, tensor2)}")
    print(f"   First tensor:\n{tensor1}")
    
    print("\n5. Testing Different Seeds")
    
    set_seed(42)
    result1 = torch.randn(5)
    
    set_seed(123)
    result2 = torch.randn(5)
    
    print(f"   Seed 42: {result1}")
    print(f"   Seed 123: {result2}")
    print(f"   Different results: {not torch.allclose(result1, result2)}")
    

    print("\n6. Testing cuDNN Settings")
    set_seed(42)
    print(f"   cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print("\n7. Testing with DataLoader")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dataset
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(data, labels)
    
    # Create loaders with same seed
    set_seed(42)
    loader1 = DataLoader(dataset, batch_size=10, shuffle=True)
    batch1_indices = []
    
    for batch in loader1:
        batch1_indices.append(batch[1][:3].tolist())  # First 3 labels
        break
    
    set_seed(42)
    loader2 = DataLoader(dataset, batch_size=10, shuffle=True)
    batch2_indices = []
    
    for batch in loader2:
        batch2_indices.append(batch[1][:3].tolist())
        break
    
    print(f"   First loader batch: {batch1_indices}")
    print(f"   Second loader batch: {batch2_indices}")
    print(f"   Same batches: {batch1_indices == batch2_indices}")
    
    print("\n Seed tests complete!")
    print("\nNote: For full reproducibility in training:")
    print("  1. Set seed before creating model, dataloaders, and training")
    print("  2. Use set_seed() or make_deterministic()")
    print("  3. Be aware that GPU operations may still have small variations")
    print("  4. Use num_workers=0 in DataLoader for perfect reproducibility")
