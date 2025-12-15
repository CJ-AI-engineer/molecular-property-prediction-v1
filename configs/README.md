# Configuration Files

This directory contains ready-to-use configuration files for training molecular property prediction models.

## Quick Start

### Basic Usage

```bash
# Train GCN on BBBP dataset
python scripts/train.py --config configs/gcn_bbbp.yaml

# Train GIN on HIV dataset
python scripts/train.py --config configs/gin_hiv.yaml

# Train GAT on BBBP dataset
python scripts/train.py --config configs/gat_bbbp.yaml
```

### Override Config Values

You can override any config value from the command line:

```bash
# Change learning rate
python scripts/train.py --config configs/gcn_bbbp.yaml --lr 0.0005

# Use different dataset
python scripts/train.py --config configs/gcn_bbbp.yaml --dataset HIV

# Enable W&B tracking
python scripts/train.py --config configs/gcn_bbbp.yaml --use-wandb

# Use different GPU
python scripts/train.py --config configs/gcn_bbbp.yaml --gpu 1
```

## Available Configurations

### Model-Specific Configs

| Config File | Model | Dataset | Task | Description |
|-------------|-------|---------|------|-------------|
| `gcn_bbbp.yaml` | GCN | BBBP | Classification | Blood-brain barrier penetration |
| `gin_hiv.yaml` | GIN | HIV | Classification | HIV inhibition activity |
| `gat_bbbp.yaml` | GAT | BBBP | Classification | With attention mechanism |
| `gcn_esol.yaml` | GCN | ESOL | Regression | Aqueous solubility |
| `gin_freesolv.yaml` | GIN | FreeSolv | Regression | Hydration free energy |

### Special Configs

| Config File | Purpose | Use Case |
|-------------|---------|----------|
| `default.yaml` | Template | Reference for all options |
| `fast_train.yaml` | Quick experiments | Testing, debugging, prototyping |
| `gin_highperf.yaml` | Best performance | Production, benchmarking |
| `ensemble.yaml` | Uncertainty | Uncertainty quantification |

## Configuration Sections

### Model Configuration

```yaml
model:
  type: gcn              # Model architecture
  hidden_dim: 128        # Hidden layer size
  num_layers: 5          # Number of GNN layers
  dropout: 0.1           # Dropout probability
  pooling: mean          # Pooling method
```

**Model Types**:
- `gcn` - Graph Convolutional Network (fast, simple)
- `gin` - Graph Isomorphism Network (most expressive)
- `gat` - Graph Attention Network (interpretable)
- `gin_edge` - GIN with edge features

**Pooling Methods**:
- `mean` - Average pooling (default)
- `add` - Sum pooling (recommended for GIN)
- `max` - Max pooling
- `attention` - Attention-based pooling
- `set2set` - Set2Set pooling

### Data Configuration

```yaml
data:
  dataset: BBBP          # Dataset name
  batch_size: 32         # Batch size
  split_type: scaffold   # Splitting strategy
  seed: 42               # Random seed
```

**Datasets**:
- `BBBP` - Blood-brain barrier (2,050 molecules, classification)
- `HIV` - HIV inhibition (41,127 molecules, classification)
- `ESOL` - Solubility (1,128 molecules, regression)
- `FreeSolv` - Hydration free energy (642 molecules, regression)
- `QM9` - Quantum properties (130,831 molecules, regression)

**Split Types**:
- `random` - Random split
- `scaffold` - Scaffold-based (prevents data leakage)
- `stratified` - Maintains class balance
- `temporal` - Time-based split

### Training Configuration

```yaml
training:
  task_type: classification  # Task type
  epochs: 100                # Max epochs
  patience: 20               # Early stopping
  gradient_clip: 1.0         # Gradient clipping
```

### Optimizer Configuration

```yaml
optimizer:
  type: adam            # Optimizer
  lr: 0.001             # Learning rate
  weight_decay: 1e-5    # L2 regularization
```

**Optimizers**:
- `adam` - Adam (default, recommended)
- `adamw` - AdamW (better weight decay)
- `sgd` - SGD with momentum
- `rmsprop` - RMSprop

### Scheduler Configuration

```yaml
scheduler:
  type: reduce_on_plateau  # Scheduler type
  factor: 0.5              # LR reduction factor
  patience: 10             # Patience
```

**Schedulers**:
- `reduce_on_plateau` - Reduce when metric plateaus (recommended)
- `step` - Step decay
- `exponential` - Exponential decay
- `cosine` - Cosine annealing
- `one_cycle` - One cycle policy
- `none` - No scheduler

### Experiment Configuration

```yaml
experiment:
  name: my_experiment    # Experiment name
  save_dir: ./checkpoints
  log_dir: ./logs
  use_wandb: true        # Enable W&B
  seed: 42
```

## Creating Custom Configs

1. **Copy a template**:
   ```bash
   cp configs/gcn_bbbp.yaml configs/my_config.yaml
   ```

2. **Edit the config**:
   ```yaml
   model:
     hidden_dim: 256  # Increase model capacity
   
   training:
     epochs: 200      # Train longer
   
   experiment:
     name: my_custom_experiment
   ```

3. **Run training**:
   ```bash
   python scripts/train.py --config configs/my_config.yaml
   ```

## Best Practices

### For Classification Tasks

```yaml
model:
  type: gin              # GIN for best performance
  hidden_dim: 256
  pooling: add           # Sum pooling for GIN

data:
  split_type: scaffold   # Prevent data leakage

scheduler:
  type: reduce_on_plateau
  mode: max              # Maximize ROC-AUC
```

### For Regression Tasks

```yaml
model:
  type: gcn              # GCN is sufficient
  hidden_dim: 128

scheduler:
  type: reduce_on_plateau
  mode: min              # Minimize RMSE

loss:
  type: mse              # Or huber for robustness
```

### For Small Datasets (<1000 samples)

```yaml
model:
  hidden_dim: 64         # Smaller model
  dropout: 0.2           # Higher dropout

training:
  epochs: 300            # More epochs
  patience: 50

optimizer:
  weight_decay: 1e-4     # Stronger regularization
```

### For Large Datasets (>10,000 samples)

```yaml
model:
  hidden_dim: 512        # Larger model
  num_layers: 7

data:
  batch_size: 128        # Larger batches

optimizer:
  type: adamw
  lr: 0.0003
```

## Hyperparameter Tuning

Use these configs as starting points for hyperparameter search:

```bash
# Search around a config
python scripts/hyperparameter_search.py \
    --config configs/gcn_bbbp.yaml \
    --n-trials 50
```

## Expected Performance

### BBBP (Classification)
- **GCN**: ROC-AUC ~0.88-0.92
- **GIN**: ROC-AUC ~0.90-0.94
- **GAT**: ROC-AUC ~0.89-0.93

### HIV (Classification)
- **GCN**: ROC-AUC ~0.75-0.80
- **GIN**: ROC-AUC ~0.78-0.82
- **GAT**: ROC-AUC ~0.76-0.81

### ESOL (Regression)
- **GCN**: RMSE ~0.5-0.7 log units
- **GIN**: RMSE ~0.4-0.6 log units

### FreeSolv (Regression)
- **GCN**: RMSE ~1.0-1.5 kcal/mol
- **GIN**: RMSE ~0.8-1.2 kcal/mol

*Note: Performance varies with random seed and hyperparameters*

## Troubleshooting

### Training is slow
- Use `configs/fast_train.yaml`
- Increase `batch_size`
- Reduce `num_layers` or `hidden_dim`
- Use `num_workers: 0` if data loading is bottleneck

### Out of memory
- Reduce `batch_size`
- Reduce `hidden_dim`
- Reduce `num_layers`
- Use gradient checkpointing (advanced)

### Poor performance
- Try different `split_type` (scaffold is harder)
- Increase `hidden_dim` or `num_layers`
- Adjust `learning_rate`
- Train longer (increase `epochs` and `patience`)
- Try different model (`gin` usually best)

### Model not converging
- Reduce `learning_rate`
- Add/increase `weight_decay`
- Use `gradient_clip`
- Check data preprocessing

## References

- **GCN**: Kipf & Welling (2017) - Semi-Supervised Classification with Graph Convolutional Networks
- **GIN**: Xu et al. (2019) - How Powerful are Graph Neural Networks?
- **GAT**: Veličković et al. (2018) - Graph Attention Networks
- **MoleculeNet**: Wu et al. (2018) - MoleculeNet: A Benchmark for Molecular Machine Learning
