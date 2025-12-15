"""
Baseline models using classical molecular descriptors.
For comparison with GNN models - important to show GNNs actually help!
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional, Literal

from .base_model import BaselineModel


class MLPBaseline(BaselineModel):
    """
    Multi-layer Perceptron baseline.
    Uses classical molecular descriptors (Morgan fingerprints, etc.)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        num_tasks: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize MLP baseline.
        
        Args:
            input_dim: Input feature dimension (e.g., 2048 for Morgan FP)
            hidden_dims: List of hidden layer dimensions
            num_tasks: Number of prediction tasks
            dropout: Dropout probability
        """
        super().__init__(input_dim, hidden_dims, num_tasks, dropout)


class RandomForestBaseline:
    """
    Random Forest baseline.
    Simple but often strong baseline for molecular property prediction.
    """
    
    def __init__(
        self,
        task_type: Literal['classification', 'regression'] = 'classification',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Random Forest baseline.
        
        Args:
            task_type: 'classification' or 'regression'
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            random_state: Random seed
            **kwargs: Additional sklearn RandomForest parameters
        """
        self.task_type = task_type
        self.random_state = random_state
        
        if task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model.
        
        Args:
            X: Features [n_samples, n_features]
            y: Targets [n_samples]
        """
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            predictions: [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (classification only).
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            probabilities: [n_samples, n_classes]
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances.
        
        Returns:
            importances: [n_features]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.feature_importances_


class SVMBaseline:
    """
    Support Vector Machine baseline.
    Can work well with high-dimensional molecular descriptors.
    """
    
    def __init__(
        self,
        task_type: Literal['classification', 'regression'] = 'classification',
        kernel: str = 'rbf',
        C: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize SVM baseline.
        
        Args:
            task_type: 'classification' or 'regression'
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            random_state: Random seed
            **kwargs: Additional sklearn SVM parameters
        """
        self.task_type = task_type
        self.random_state = random_state
        
        if task_type == 'classification':
            self.model = SVC(
                kernel=kernel,
                C=C,
                random_state=random_state,
                probability=True,  # Enable probability estimates
                **kwargs
            )
        else:
            self.model = SVR(
                kernel=kernel,
                C=C,
                **kwargs
            )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline.
    Fast and interpretable - good sanity check.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Logistic Regression baseline.
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
            **kwargs: Additional sklearn parameters
        """
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RidgeBaseline:
    """
    Ridge Regression baseline.
    Linear model with L2 regularization.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        **kwargs
    ):
        """
        Initialize Ridge Regression baseline.
        
        Args:
            alpha: Regularization strength
            **kwargs: Additional sklearn parameters
        """
        self.model = Ridge(alpha=alpha, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


if __name__ == "__main__":
    print("Testing baseline models...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 2048  # Morgan fingerprint size
    
    X_train = np.random.randn(n_samples, n_features)
    y_class = np.random.randint(0, 2, n_samples) 
    y_reg = np.random.randn(n_samples)  
    
    X_test = np.random.randn(20, n_features)
    
    # Test 1: MLP Baseline
    print("\n1. Testing MLP Baseline")
    mlp = MLPBaseline(
        input_dim=n_features,
        hidden_dims=[512, 256, 128],
        num_tasks=1,
        dropout=0.1
    )
    
    print(f"Parameters: {mlp.count_parameters():,}")
    
    # Test forward pass
    X_tensor = torch.FloatTensor(X_test)
    output = mlp(X_tensor)
    print(f"Output shape: {output.shape}")
    
    # Test 2: Random Forest Classification
    print("\n2. Testing Random Forest (Classification)")
    rf_clf = RandomForestBaseline(
        task_type='classification',
        n_estimators=100,
        max_depth=10
    )
    
    rf_clf.fit(X_train, y_class)
    preds = rf_clf.predict(X_test)
    probs = rf_clf.predict_proba(X_test)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Prediction range: [{preds.min()}, {preds.max()}]")
    
    importances = rf_clf.get_feature_importance()
    print(f"Feature importances shape: {importances.shape}")
    print(f"Top 5 important features: {np.argsort(importances)[-5:]}")
    
    # Test 3: Random Forest Regression
    print("\n3. Testing Random Forest (Regression)")
    rf_reg = RandomForestBaseline(
        task_type='regression',
        n_estimators=100
    )
    
    rf_reg.fit(X_train, y_reg)
    preds = rf_reg.predict(X_test)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Prediction range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    # Test 4: SVM Classification
    print("\n4. Testing SVM (Classification)")
    svm_clf = SVMBaseline(
        task_type='classification',
        kernel='rbf',
        C=1.0
    )
    
    svm_clf.fit(X_train, y_class)
    preds = svm_clf.predict(X_test)
    probs = svm_clf.predict_proba(X_test)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Test 5: SVM Regression
    print("\n5. Testing SVM (Regression)")
    svm_reg = SVMBaseline(
        task_type='regression',
        kernel='rbf',
        C=1.0
    )
    
    svm_reg.fit(X_train, y_reg)
    preds = svm_reg.predict(X_test)
    
    print(f"Predictions shape: {preds.shape}")
    
    # Test 6: Logistic Regression
    print("\n6. Testing Logistic Regression")
    lr = LogisticRegressionBaseline(C=1.0)
    
    lr.fit(X_train, y_class)
    preds = lr.predict(X_test)
    probs = lr.predict_proba(X_test)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Test 7: Ridge Regression
    print("\n7. Testing Ridge Regression")
    ridge = RidgeBaseline(alpha=1.0)
    
    ridge.fit(X_train, y_reg)
    preds = ridge.predict(X_test)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Prediction range: [{preds.min():.3f}, {preds.max():.3f}]")
    
    print("\n Baseline model tests complete!")
