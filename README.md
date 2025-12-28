# Assessing the Impact of Structural Learning Algorithms on Generative Model Rankings

A modular research framework for evaluating how different causal structure learning algorithms affect the ranking and evaluation of tabular generative models. This project implements the CauTabBench evaluation methodology and supports multiple state-of-the-art generative models and structure learning algorithms.

## Project Overview

This framework addresses a key research question: How do structural learning algorithms impact the evaluation and ranking of generative models when the true causal structure is unknown?

The methodology:
1. Train multiple generative models (CTGAN, TabDDPM, GMM, Bayesian Network) on real data
2. Generate synthetic data from each model
3. Learn causal structures from data using various algorithms (PC, GES, NOTEARS, FCI, LiNGAM)
4. Evaluate models using CauTabBench methodology with both ground truth and learned structures
5. Compare how model rankings change across different structure learning methods

## Repository Structure

```
.
├── benchmarks_with_ground_truth/  # Input datasets
│   ├── csv/                       # CSV data files
│   └── txt/                       # Ground truth causal structures (edge lists)
├── config/                        # Configuration files
│   ├── datasets.yaml              # Dataset configurations
│   ├── models.yaml                # Model hyperparameters
│   └── structure_learning.yaml   # Structure learning algorithm configs
├── data/                          # Processed data
│   ├── preprocessed/              # Tab-ddpm format datasets
│   └── info/                      # Dataset metadata
├── src/                           # Source code
│   ├── data/                      # Data loading and preprocessing
│   ├── models/                    # Generative model wrappers
│   ├── structure_learning/        # Causal discovery algorithms
│   ├── evaluation/                # CauTabBench evaluation
│   ├── tuning/                    # Optuna hyperparameter tuning
│   └── utils/                     # Utility functions
├── scripts/                       # CLI scripts for workflow
│   ├── 01_preprocess_data.py      # Preprocess datasets
│   ├── 02_train_model.py          # Train generative models
│   ├── 03_tune_model.py           # Hyperparameter tuning
│   ├── 04_generate_synthetic.py   # Generate synthetic data
│   ├── 05_learn_structure.py      # Learn causal structures
│   ├── 06_evaluate.py             # Evaluate models
│   └── 07_compare_rankings.py     # Compare rankings
├── outputs/                       # Generated outputs
│   ├── models/                    # Trained models
│   ├── synthetic/                 # Synthetic datasets
│   ├── structures/                # Learned causal graphs
│   ├── evaluations/               # Evaluation results
│   └── rankings/                  # Ranking comparisons
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended for TabDDPM and NOTEARS)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shahdhammoud/Assessing-the-impact-of-structural-learning-algorithms-on-the-results-of-comparing-generative-models.git
cd Assessing-the-impact-of-structural-learning-algorithms-on-the-results-of-comparing-generative-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your datasets:
   - Place CSV files in `benchmarks_with_ground_truth/csv/`
   - Place corresponding TXT structure files in `benchmarks_with_ground_truth/txt/`

### Dataset Format

**CSV Files:**
- Must have header row with column names
- First column may be an index (will be auto-detected and removed)
- Last column is the target feature
- Can contain both categorical and continuous features

**TXT Files:**
- Each line represents a directed edge: `source target`
- Example:
  ```
  A C
  B C
  D E
  ```
  This represents edges: A→C, B→C, D→E

## Usage

### Complete Workflow Example

```bash
# 1. Preprocess dataset
python scripts/01_preprocess_data.py --dataset alarm

# 2. Tune hyperparameters (optional but recommended)
python scripts/03_tune_model.py --dataset alarm --model ctgan --trials 100
python scripts/03_tune_model.py --dataset alarm --model gmm --trials 50

# 3. Train models with best parameters
python scripts/02_train_model.py --dataset alarm --model ctgan --use-best-params
python scripts/02_train_model.py --dataset alarm --model gmm --use-best-params
python scripts/02_train_model.py --dataset alarm --model bayesian_network

# 4. Generate synthetic data
python scripts/04_generate_synthetic.py --dataset alarm --model ctgan --n-samples 1000
python scripts/04_generate_synthetic.py --dataset alarm --model gmm --n-samples 1000
python scripts/04_generate_synthetic.py --dataset alarm --model bayesian_network --n-samples 1000

# 5. Learn structures from synthetic data
python scripts/05_learn_structure.py --dataset alarm --algorithm pc --data-type synthetic --model ctgan
python scripts/05_learn_structure.py --dataset alarm --algorithm ges --data-type synthetic --model ctgan
python scripts/05_learn_structure.py --dataset alarm --algorithm notears --data-type synthetic --model gmm

# 6. Evaluate models with different structures
python scripts/06_evaluate.py --dataset alarm --model ctgan --structure ground_truth
python scripts/06_evaluate.py --dataset alarm --model ctgan --structure pc_learned_synthetic_ctgan
python scripts/06_evaluate.py --dataset alarm --model gmm --structure ground_truth
python scripts/06_evaluate.py --dataset alarm --model gmm --structure ges_learned_synthetic_gmm

# 7. Compare rankings across structures
python scripts/07_compare_rankings.py --dataset alarm --structures ground_truth,pc_learned,ges_learned,notears_learned
```

### Quick Start Commands

```bash
# Preprocess all datasets
for dataset in dataset1 dataset2 dataset3; do
    python scripts/01_preprocess_data.py --dataset $dataset
done

# Train all models on a dataset
for model in ctgan gmm bayesian_network; do
    python scripts/02_train_model.py --dataset alarm --model $model
done

# Learn structures with all algorithms
for algo in pc ges notears fci lingam; do
    python scripts/05_learn_structure.py --dataset alarm --algorithm $algo --data-type real
done
```

## Supported Models

1. **CTGAN**: Conditional Tabular GAN (uses GPU if available)
2. **TabDDPM**: Denoising Diffusion Probabilistic Model for tabular data (requires GPU, integration pending)
3. **GMM**: Gaussian Mixture Model
4. **Bayesian Network**: Generative Bayesian Network with structure learning

## Supported Structure Learning Algorithms

| Algorithm | Description | GPU Support | Library |
|-----------|-------------|-------------|---------|
| **PC** | Peter-Clark constraint-based | No | causal-learn |
| **GES** | Greedy Equivalence Search | No | causal-learn |
| **NOTEARS** | Continuous optimization for DAGs | Yes | Custom PyTorch |
| **FCI** | Fast Causal Inference (handles latent confounders) | No | causal-learn |
| **LiNGAM** | Linear Non-Gaussian Acyclic Model | No | lingam |

## Evaluation Methodology

The framework implements the CauTabBench evaluation approach:

1. Extract conditional independence (CI) constraints from causal structure
2. Test these CI relationships on both real and synthetic data
3. Compute agreement rate between real and synthetic CI tests
4. Calculate violation rates for both datasets
5. Generate overall quality score combining agreement and violations

**Quality Score** = 0.6 × (Agreement Rate) + 0.4 × (1 - |Real Violation - Synthetic Violation|)

Higher scores indicate better synthetic data quality.

## Configuration

### Model Hyperparameters
Edit `config/models.yaml` to customize:
- Default parameters for each model
- Hyperparameter search spaces for tuning
- Training settings (epochs, batch size, learning rate, etc.)

### Dataset Settings
Edit `config/datasets.yaml` to configure:
- Categorical threshold for automatic type detection
- Missing value strategies
- Train/test split ratios

### Structure Learning
Edit `config/structure_learning.yaml` to adjust:
- Algorithm-specific parameters
- Independence test methods
- Significance levels

## GPU Support

### Enabling CUDA

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Set GPU device
export CUDA_VISIBLE_DEVICES=0
```

Models that benefit from GPU:
- CTGAN: Automatically uses GPU if available
- TabDDPM: Requires GPU
- NOTEARS: Can use GPU for acceleration

## Output Files

After running the complete workflow, you'll have:

- **Trained Models**: `outputs/models/{dataset}_{model}.pkl`
- **Synthetic Data**: `outputs/synthetic/{dataset}_{model}_synthetic.csv`
- **Learned Structures**: `outputs/structures/{dataset}_{algorithm}_{suffix}.json`
- **Structure Visualizations**: `outputs/structures/{dataset}_{algorithm}_{suffix}.png`
- **Evaluation Results**: `outputs/evaluations/{dataset}_{model}_{structure}_eval.json`
- **Ranking Comparisons**: `outputs/rankings/{dataset}_ranking_comparison.json`

## Key Features

- **Modular Design**: Each step is independent, run in any order
- **Flexible Configuration**: YAML configs for easy experimentation
- **Automatic Type Detection**: Identifies categorical vs continuous features
- **Comprehensive Evaluation**: CauTabBench methodology implementation
- **Statistical Comparison**: Kendall's tau and Spearman's rho for ranking correlation
- **Hyperparameter Optimization**: Optuna integration for systematic tuning
- **GPU Acceleration**: CUDA support where applicable

## Extending the Framework

### Adding a New Generative Model

1. Create wrapper in `src/models/your_model.py` inheriting from `BaseGenerativeModel`
2. Implement `fit()` and `sample()` methods
3. Add configuration to `config/models.yaml`
4. Register in `scripts/02_train_model.py`

### Adding a New Structure Learning Algorithm

1. Create implementation in `src/structure_learning/your_algo.py` inheriting from `BaseStructureLearner`
2. Implement `fit()` method returning `nx.DiGraph`
3. Add configuration to `config/structure_learning.yaml`
4. Register in `scripts/05_learn_structure.py`

## Troubleshooting

**Issue**: Out of memory errors
- Solution: Reduce batch size in model configs or use smaller datasets

**Issue**: Structure learning takes too long
- Solution: Reduce `max_iter` or `max_cond_vars` in algorithm configs

**Issue**: TabDDPM not working
- Note: TabDDPM integration is pending. Use CTGAN, GMM, or Bayesian Network instead.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{structural_learning_impact,
  title={Framework for Assessing Structural Learning Impact on Generative Model Rankings},
  author={Your Name},
  year={2024},
  url={https://github.com/shahdhammoud/Assessing-the-impact-of-structural-learning-algorithms-on-the-results-of-comparing-generative-models}
}
```

## References

- CauTabBench: https://github.com/TURuibo/CauTabBench
- tab-ddpm: https://github.com/yandex-research/tab-ddpm
- causal-learn: https://github.com/py-why/causal-learn
- CTGAN/SDV: https://github.com/sdv-dev/SDV

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
