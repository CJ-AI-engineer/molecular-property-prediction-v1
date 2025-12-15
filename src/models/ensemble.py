"""
Deep Ensemble for uncertainty quantification.
Trains multiple models with different initializations.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Type
from torch_geometric.data import Data
from .base_model import BaseMolecularModel
import tempfile
import os


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification.
    
    Trains multiple instances of the same model with different
    random initializations. Predictions are averaged, and variance
    provides uncertainty estimates.
    
    Reference: Lakshminarayanan et al. (2017) "Simple and Scalable
    Predictive Uncertainty Estimation using Deep Ensembles"
    """
    
    def __init__(
        self,
        model_class: Type[BaseMolecularModel],
        model_kwargs: dict,
        n_models: int = 5
    ):
        """
        Initialize deep ensemble.
        
        Args:
            model_class: Class of model to ensemble (e.g., MolecularGCN)
            model_kwargs: Arguments to pass to model constructor
            n_models: Number of models in ensemble
        """
        super().__init__()
        
        self.n_models = n_models
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            model_class(**model_kwargs)
            for _ in range(n_models)
        ])
    
    def forward(
        self,
        data: Data,
        return_individual: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            data: PyTorch Geometric Data object
            return_individual: Whether to return individual predictions
            
        Returns:
            If return_individual=False:
                mean_prediction: [batch_size, num_tasks]
            If return_individual=True:
                all_predictions: [n_models, batch_size, num_tasks]
        """
        predictions = []
        
        for model in self.models:
            pred = model(data)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        if return_individual:
            return predictions
        else:
            return predictions.mean(dim=0)
    
    def predict_with_uncertainty(
        self,
        data: Data
    ) -> tuple:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Tuple of (mean, variance, std)
                - mean: [batch_size, num_tasks]
                - variance: [batch_size, num_tasks]
                - std: [batch_size, num_tasks]
        """
        predictions = self.forward(data, return_individual=True)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        std = predictions.std(dim=0)
        
        return mean, variance, std
    
    def get_model(self, idx: int) -> BaseMolecularModel:
        """
        Get individual model from ensemble.
        
        Args:
            idx: Model index
            
        Returns:
            Model at index
        """
        return self.models[idx]
    
    def count_parameters(self) -> int:
        """Count total parameters across all models."""
        return sum(
            model.count_parameters()
            for model in self.models
        )
    
    def save_ensemble(self, path_template: str, **kwargs):
        """
        Save all models in ensemble.
        
        Args:
            path_template: Path template with {} for model index
                          e.g., "models/ensemble_{}.pt"
            **kwargs: Additional items to save
        """
        for i, model in enumerate(self.models):
            path = path_template.format(i)
            model.save_checkpoint(path, **kwargs)
    
    @classmethod
    def load_ensemble(
        cls,
        path_template: str,
        n_models: int,
        model_class: Type[BaseMolecularModel],
        map_location: Optional[str] = None
    ):
        """
        Load ensemble from saved checkpoints.
        
        Args:
            path_template: Path template with {} for model index
            n_models: Number of models
            model_class: Model class
            map_location: Device to load to
            
        Returns:
            DeepEnsemble instance
        """
        path0 = path_template.format(0)
        model0, checkpoint = model_class.load_checkpoint(path0, map_location)
        
        config = model0.get_config()
        config.pop('model_class', None)
        
        ensemble = cls(
            model_class=model_class,
            model_kwargs=config,
            n_models=n_models
        )
        
        for i in range(n_models):
            path = path_template.format(i)
            model, _ = model_class.load_checkpoint(path, map_location)
            ensemble.models[i] = model
        
        return ensemble, checkpoint



if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    from .gcn import MolecularGCN
    
    print("Testing DeepEnsemble...")
    
    # Create sample data
    x = torch.randn(6, 50)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)
    
    data1 = Data(x=x, edge_index=edge_index)
    data1.batch = torch.zeros(6, dtype=torch.long)
    
    x2 = torch.randn(4, 50)
    edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    
    data2 = Data(x=x2, edge_index=edge_index2)
    data2.batch = torch.ones(4, dtype=torch.long)
    
    batch_data = Batch.from_data_list([data1, data2])
    
    # Test 1: Create ensemble
    print("\n1. Creating Deep Ensemble")
    
    model_kwargs = {
        'node_feat_dim': 50,
        'edge_feat_dim': 10,
        'hidden_dim': 64,
        'num_tasks': 1,
        'num_layers': 3,
        'dropout': 0.1
    }
    
    ensemble = DeepEnsemble(
        model_class=MolecularGCN,
        model_kwargs=model_kwargs,
        n_models=5
    )
    
    print(f"Number of models: {ensemble.n_models}")
    print(f"Total parameters: {ensemble.count_parameters():,}")
    print(f"Parameters per model: {ensemble.get_model(0).count_parameters():,}")
    
    # Test 2: Forward pass
    print("\n2. Testing Forward Pass")
    ensemble.eval()
    
    with torch.no_grad():
        # Mean prediction
        mean_pred = ensemble(batch_data)
        print(f"Mean prediction shape: {mean_pred.shape}")
        
        # Individual predictions
        individual_preds = ensemble(batch_data, return_individual=True)
        print(f"Individual predictions shape: {individual_preds.shape}")
    
    # Test 3: Uncertainty estimation
    print("\n3. Testing Uncertainty Estimation")
    
    with torch.no_grad():
        mean, variance, std = ensemble.predict_with_uncertainty(batch_data)
    
    print(f"Mean shape: {mean.shape}")
    print(f"Variance shape: {variance.shape}")
    print(f"Std shape: {std.shape}")
    
    print(f"\nExample predictions:")
    print(f"  Mean: {mean[0].item():.4f}")
    print(f"  Std: {std[0].item():.4f}")
    print(f"  95% CI: [{mean[0].item() - 2*std[0].item():.4f}, "
          f"{mean[0].item() + 2*std[0].item():.4f}]")
    
    # Test 4: Individual model access
    print("\n4. Testing Individual Model Access")
    
    model0 = ensemble.get_model(0)
    print(f"Model 0 type: {type(model0).__name__}")
    
    with torch.no_grad():
        pred0 = model0(batch_data)
    
    print(f"Model 0 prediction: {pred0[0].item():.4f}")
    
    # Test 5: Gradient flow
    print("\n5. Testing Gradient Flow")
    
    ensemble.train()
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
    
    output = ensemble(batch_data)
    loss = output.mean()
    loss.backward()
    
    # Check if all models have gradients
    all_have_grads = all(
        any(p.grad is not None for p in model.parameters())
        for model in ensemble.models
    )
    
    print(f"All models have gradients: {all_have_grads}")
    
    optimizer.step()
    print(f"Optimizer step successful")
    
    # Test 6: Save/load ensemble
    print("\n6. Testing Save/Load")
    

    
    with tempfile.TemporaryDirectory() as tmpdir:
        path_template = os.path.join(tmpdir, "ensemble_{}.pt")
        
        ensemble.save_ensemble(path_template, epoch=10, best_loss=0.3)
        print(f"Saved ensemble to {tmpdir}")
        
        loaded_ensemble, checkpoint = DeepEnsemble.load_ensemble(
            path_template,
            n_models=5,
            model_class=MolecularGCN
        )
        
        print(f"Loaded ensemble")
        print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
        
        with torch.no_grad():
            loaded_pred = loaded_ensemble(batch_data)
        
        print(f"Loaded ensemble prediction: {loaded_pred[0].item():.4f}")
    
    print("\n7. Uncertainty Across Models")
    
    ensemble.eval()
    with torch.no_grad():
        individual = ensemble(batch_data, return_individual=True)
    
    print(f"Individual predictions for first sample:")
    for i in range(ensemble.n_models):
        print(f"  Model {i}: {individual[i, 0].item():.4f}")
    
    print(f"  Mean: {individual[:, 0].mean().item():.4f}")
    print(f"  Std: {individual[:, 0].std().item():.4f}")
    
    print("\n Deep Ensemble tests complete!")
