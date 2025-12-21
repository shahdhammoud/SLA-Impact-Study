# Quick Start Guide - New Features

This guide provides quick examples for using the newly implemented features.

## 1. Using TabDDPM Model

### Basic Training
```python
from src.models.tabddpm_wrapper import TabDDPMWrapper
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize model
model = TabDDPMWrapper(
    epochs=1000,           # Number of training epochs
    batch_size=1024,       # Batch size for training
    num_timesteps=1000,    # Diffusion timesteps
    hidden_dim=256,        # Hidden layer dimension
    num_layers=4,          # Number of MLP layers
    lr=0.002,             # Learning rate
    use_gpu=True          # Use GPU if available
)

# Train the model
model.fit(data)

# Generate synthetic samples
synthetic_data = model.sample(n_samples=1000)

# Get training losses
losses = model.get_training_losses()
print(f"Final loss: {losses[-1]:.4f}")
```

### Training with Command Line
```bash
python scripts/02_train_model.py \
    --dataset sachs \
    --model tabddpm \
    --config config/models.yaml \
    --data-dir data/preprocessed \
    --output-dir outputs/models
```

This will automatically:
- Train the TabDDPM model
- Save the trained model to `outputs/models/sachs_tabddpm.pkl`
- Generate and save training loss plot to `outputs/models/sachs_tabddpm_training_loss.png`
- Save raw losses to `outputs/models/sachs_tabddpm_training_losses.json`

## 2. Training Loss Visualization

### Plotting Single Model Loss
```python
from src.utils.visualization_utils import plot_training_loss

losses = [1.5, 1.2, 1.0, 0.9, 0.85, 0.8, 0.78, 0.76, 0.75, 0.74]

plot_training_loss(
    losses=losses,
    model_name='TabDDPM',
    save_path='outputs/tabddpm_loss.png',
    title='TabDDPM Training Loss'
)
```

### Comparing Multiple Models
```python
from src.utils.visualization_utils import plot_multiple_training_losses

losses_dict = {
    'TabDDPM': tabddpm_losses,
    'CTGAN': ctgan_losses
}

plot_multiple_training_losses(
    losses_dict=losses_dict,
    save_path='outputs/comparison.png',
    title='Training Loss Comparison'
)
```

## 3. SHD Percentage Calculation

### Computing SHD
```python
from src.utils.graph_utils import compute_shd, compute_graph_metrics
import networkx as nx

# Create true and learned graphs
true_graph = nx.DiGraph()
true_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

learned_graph = nx.DiGraph()
learned_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('D', 'C')])

# Get SHD as percentage (default)
shd_pct = compute_shd(true_graph, learned_graph, return_percentage=True)
print(f"SHD: {shd_pct:.2f}%")

# Get absolute SHD if needed
shd_abs = compute_shd(true_graph, learned_graph, return_percentage=False)
print(f"SHD (absolute): {shd_abs}")

# Get full metrics
metrics = compute_graph_metrics(true_graph, learned_graph)
print(f"SHD: {metrics['shd']:.2f}%")  # percentage
print(f"SHD (absolute): {metrics['shd_absolute']}")  # absolute value
print(f"F1 Score: {metrics['f1']:.3f}")
```

### Structure Learning with SHD Output
```bash
python scripts/05_learn_structure.py \
    --dataset sachs \
    --algorithm pc \
    --data-type real \
    --output-dir outputs/structures
```

Output will show:
```
Metrics: SHD=12.50% (absolute: 15), F1=0.750, Precision=0.800, Recall=0.700
```

## 4. Ranking Visualizations

### Basic Ranking Comparison
```python
from src.utils.visualization_utils import plot_model_rankings

rankings_dict = {
    'ground_truth': [('TabDDPM', 0.87), ('CTGAN', 0.82), ('GMM', 0.76)],
    'pc_learned': [('CTGAN', 0.84), ('TabDDPM', 0.83), ('GMM', 0.74)],
    'ges_learned': [('TabDDPM', 0.85), ('CTGAN', 0.81), ('GMM', 0.77)]
}

plot_model_rankings(
    rankings_dict=rankings_dict,
    save_path='outputs/rankings.png',
    title='Model Rankings Across Algorithms'
)
```

### Pairwise Comparison
```python
from src.utils.visualization_utils import plot_ranking_comparison

baseline = [('TabDDPM', 0.87), ('CTGAN', 0.82), ('GMM', 0.76)]
comparison = [('CTGAN', 0.84), ('TabDDPM', 0.83), ('GMM', 0.74)]

plot_ranking_comparison(
    baseline_ranking=baseline,
    comparison_ranking=comparison,
    baseline_name='Ground Truth',
    comparison_name='PC Learned',
    save_path='outputs/comparison.png'
)
```

### Consensus Ranking
```python
from src.utils.visualization_utils import plot_consensus_ranking

# Calculate consensus (average across all structures)
consensus = [('TabDDPM', 0.85), ('CTGAN', 0.82), ('GMM', 0.76)]

plot_consensus_ranking(
    rankings_dict=rankings_dict,
    consensus_ranking=consensus,
    save_path='outputs/consensus.png'
)
```

### Full Ranking Analysis with Command Line
```bash
python scripts/07_compare_rankings.py \
    --dataset sachs \
    --structures ground_truth,pc_learned,ges_learned,notears_learned \
    --models ctgan,gmm,bayesian_network,tabddpm \
    --eval-dir outputs/evaluations \
    --output-dir outputs/rankings \
    --baseline ground_truth
```

This generates:
- `outputs/rankings/sachs_rankings_overview.png` - Overview heatmap and bars
- `outputs/rankings/sachs_ranking_ground_truth_vs_*.png` - Pairwise comparisons
- `outputs/rankings/sachs_consensus_ranking.png` - Consensus ranking
- `outputs/rankings/sachs_ranking_comparison.json` - Statistical comparison metrics

## 5. Complete Pipeline Example

### Running Full Analysis
```bash
# 1. Preprocess data
python scripts/01_preprocess_data.py --dataset sachs

# 2. Train models (with automatic loss visualization)
python scripts/02_train_model.py --dataset sachs --model tabddpm
python scripts/02_train_model.py --dataset sachs --model ctgan
python scripts/02_train_model.py --dataset sachs --model gmm

# 3. Generate synthetic data
python scripts/04_generate_synthetic.py --dataset sachs --model tabddpm
python scripts/04_generate_synthetic.py --dataset sachs --model ctgan
python scripts/04_generate_synthetic.py --dataset sachs --model gmm

# 4. Learn structures (shows SHD as percentage)
python scripts/05_learn_structure.py --dataset sachs --algorithm pc --data-type real
python scripts/05_learn_structure.py --dataset sachs --algorithm ges --data-type real

# 5. Evaluate models
python scripts/06_evaluate.py --dataset sachs --model tabddpm --structure ground_truth
python scripts/06_evaluate.py --dataset sachs --model tabddpm --structure pc_learned
# ... repeat for all models and structures

# 6. Compare rankings (with visualizations)
python scripts/07_compare_rankings.py \
    --dataset sachs \
    --structures ground_truth,pc_learned,ges_learned \
    --models tabddpm,ctgan,gmm
```

## 6. Demonstration Script

Run the comprehensive demonstration:
```bash
python3 demo_features.py
```

This will:
- Test TabDDPM training and sampling
- Generate all types of visualizations
- Show SHD percentage calculation
- Create example outputs in `outputs/demo/`

## Tips

### For Large Datasets
```python
# Reduce TabDDPM parameters for faster training
model = TabDDPMWrapper(
    epochs=500,        # Fewer epochs
    batch_size=2048,   # Larger batches
    num_timesteps=500, # Fewer timesteps
    hidden_dim=128     # Smaller network
)
```

### For Better Quality
```python
# Increase TabDDPM parameters for better quality
model = TabDDPMWrapper(
    epochs=2000,       # More epochs
    batch_size=512,    # Smaller batches
    num_timesteps=2000,# More timesteps
    hidden_dim=512,    # Larger network
    num_layers=6       # Deeper network
)
```

### Accessing Training Losses from Saved Models
```python
from src.models.tabddpm_wrapper import TabDDPMWrapper

model = TabDDPMWrapper()
model.load('outputs/models/dataset_tabddpm.pkl')
losses = model.get_training_losses()
```

## Troubleshooting

### GPU Memory Issues
If you encounter GPU memory errors:
```python
model = TabDDPMWrapper(
    batch_size=256,  # Reduce batch size
    use_gpu=False    # Use CPU instead
)
```

### Visualization Backend Issues
If plots don't display:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

## Additional Resources

- Full documentation: `IMPLEMENTATION_SUMMARY.md`
- Configuration files: `config/models.yaml`
- Example notebooks: `notebooks/`
- Test scripts: `tests/`

For more information, see the main README.md and IMPLEMENTATION_SUMMARY.md files.
