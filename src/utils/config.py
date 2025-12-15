"""
Configuration management utilities.
Load, validate, and manage experiment configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy


class Config:
    """
    Configuration manager for experiments.
    Supports YAML and JSON formats with nested access.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize config.
        
        Args:
            config_dict: Dictionary of configuration values
        """
        self._config = config_dict or {}
    
    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set config value by key."""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with default.
        
        Supports nested keys with dot notation: 'model.hidden_dim'
        
        Args:
            key: Configuration key (can use dots for nested access)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if '.' in key:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        else:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set config value.
        
        Supports nested keys with dot notation: 'model.hidden_dim'
        
        Args:
            key: Configuration key
            value: Value to set
        """
        if '.' in key:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
        else:
            self._config[key] = value
    
    def update(self, other: Union[Dict, 'Config']):
        """
        Update config with another dict or Config.
        
        Args:
            other: Dictionary or Config to merge
        """
        if isinstance(other, Config):
            other = other._config
        self._config = self._merge_dicts(self._config, other)
    
    def _merge_dicts(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return deepcopy(self._config)
    
    def save(self, filepath: Union[str, Path]):
        """
        Save config to file.
        
        Args:
            filepath: Path to save config (supports .yaml, .yml, .json)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        elif filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load config from file.
        
        Args:
            filepath: Path to config file (.yaml, .yml, .json)
            
        Returns:
            Config instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(config_dict)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"
    
    def __str__(self) -> str:
        """Pretty string representation."""
        return yaml.dump(self._config, default_flow_style=False, sort_keys=False)


def load_config(filepath: Union[str, Path]) -> Config:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Config instance
    """
    return Config.from_file(filepath)


def merge_configs(base: Config, override: Config) -> Config:
    """
    Merge two configs (override takes precedence).
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged Config
    """
    merged = Config(base.to_dict())
    merged.update(override)
    return merged


def get_default_config() -> Config:
    """
    Get default configuration for molecular property prediction.
    
    Returns:
        Default Config
    """
    default = {
        'model': {
            'type': 'gcn',
            'node_feat_dim': 50,
            'edge_feat_dim': 10,
            'hidden_dim': 128,
            'num_layers': 5,
            'dropout': 0.1,
            'pooling': 'mean',
        },
        'data': {
            'dataset': 'BBBP',
            'batch_size': 32,
            'num_workers': 4,
            'split_type': 'random',
            'split_ratios': [0.8, 0.1, 0.1],
        },
        'training': {
            'task_type': 'classification',
            'epochs': 100,
            'patience': 20,
            'gradient_clip': 1.0,
        },
        'optimizer': {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5,
        },
        'scheduler': {
            'type': 'reduce_on_plateau',
            'factor': 0.5,
            'patience': 10,
        },
        'experiment': {
            'name': 'default_experiment',
            'save_dir': './checkpoints',
            'seed': 42,
            'use_wandb': False,
            'use_mlflow': False,
        }
    }
    
    return Config(default)


def validate_config(config: Config) -> bool:
    """
    Validate configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Required keys
    required_sections = ['model', 'data', 'training', 'optimizer']
    
    for section in required_sections:
        if section not in config._config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate model config
    model_config = config.get('model', {})
    if 'type' not in model_config:
        raise ValueError("model.type is required")
    if 'hidden_dim' not in model_config:
        raise ValueError("model.hidden_dim is required")
    
    # Validate data config
    data_config = config.get('data', {})
    if 'dataset' not in data_config:
        raise ValueError("data.dataset is required")
    if 'batch_size' not in data_config:
        raise ValueError("data.batch_size is required")
    
    # Validate training config
    training_config = config.get('training', {})
    if 'task_type' not in training_config:
        raise ValueError("training.task_type is required")
    if training_config['task_type'] not in ['classification', 'regression']:
        raise ValueError("training.task_type must be 'classification' or 'regression'")
    
    return True


if __name__ == "__main__":
    import tempfile
    
    print("Testing Config utilities...")
    
    print("\n1. Testing Config Creation")
    config_dict = {
        'model': {
            'type': 'gcn',
            'hidden_dim': 128,
        },
        'training': {
            'epochs': 100,
            'lr': 0.001,
        }
    }
    
    config = Config(config_dict)
    print(f"   Created config: {config.get('model.type')}")
    
    print("\n2. Testing Nested Access")
    print(f"   model.hidden_dim: {config.get('model.hidden_dim')}")
    print(f"   training.epochs: {config.get('training.epochs')}")
    print(f"   missing.key (default=42): {config.get('missing.key', 42)}")
    
    print("\n3. Testing Set Values")
    config.set('model.dropout', 0.1)
    config.set('new_section.new_key', 'value')
    print(f"   model.dropout: {config.get('model.dropout')}")
    print(f"   new_section.new_key: {config.get('new_section.new_key')}")
    

    print("\n4. Testing Update")
    update_dict = {
        'model': {'hidden_dim': 256},
        'training': {'epochs': 200}
    }
    config.update(update_dict)
    print(f"   Updated model.hidden_dim: {config.get('model.hidden_dim')}")
    print(f"   Updated training.epochs: {config.get('training.epochs')}")
    
    print("\n5. Testing Save/Load")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / 'config.yaml'
        config.save(yaml_path)
        print(f"   Saved to {yaml_path}")
        
        loaded_config = Config.from_file(yaml_path)
        print(f"   Loaded model.type: {loaded_config.get('model.type')}")
        
        # Save as JSON
        json_path = Path(tmpdir) / 'config.json'
        config.save(json_path)
        print(f"   Saved to {json_path}")
        
        # Load from JSON
        loaded_config = Config.from_file(json_path)
        print(f"   Loaded model.type: {loaded_config.get('model.type')}")
    
    # Default config
    print("\n6. Testing Default Config")
    default = get_default_config()
    print(f"   Default model: {default.get('model.type')}")
    print(f"   Default dataset: {default.get('data.dataset')}")
    print(f"   Default epochs: {default.get('training.epochs')}")
    

    print("\n7. Testing Merge Configs")
    base = Config({'a': 1, 'b': {'c': 2, 'd': 3}})
    override = Config({'b': {'c': 10}, 'e': 5})
    merged = merge_configs(base, override)
    print(f"   Merged a: {merged.get('a')}")
    print(f"   Merged b.c: {merged.get('b.c')}")
    print(f"   Merged b.d: {merged.get('b.d')}")
    print(f"   Merged e: {merged.get('e')}")
    
    print("\n8. Testing Config Validation")
    try:
        validate_config(default)
        print("    Default config is valid")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
    
    # Test invalid config
    invalid = Config({'model': {'type': 'gcn'}})
    try:
        validate_config(invalid)
        print("   ✗ Invalid config passed validation")
    except ValueError as e:
        print(f"    Correctly caught invalid config: {e}")
    
    print("\n9. Testing Pretty Print")
    print("Config as string:")
    print(str(default))
    
    print("\n Config tests complete!")
