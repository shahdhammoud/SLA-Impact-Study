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



## References

- CauTabBench: https://github.com/TURuibo/CauTabBench
- tab-ddpm: https://github.com/yandex-research/tab-ddpm
- causal-learn: https://github.com/py-why/causal-learn
- CTGAN/SDV: https://github.com/sdv-dev/SDV
