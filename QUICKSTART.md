# Quick Start Guide

This guide will help you get started with the framework in 10 minutes.

## Prerequisites

1. Python 3.8 or higher
2. Git
3. (Optional) CUDA-capable GPU for faster training

## Installation

```bash
# Clone the repository
git clone https://github.com/shahdhammoud/Assessing-the-impact-of-structural-learning-algorithms-on-the-results-of-comparing-generative-models.git
cd Assessing-the-impact-of-structural-learning-algorithms-on-the-results-of-comparing-generative-models

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Adding Your Dataset

1. Place your CSV file in `benchmarks_with_ground_truth/csv/`
   - Example: `benchmarks_with_ground_truth/csv/mydataset.csv`
   - Ensure it has a header row with column names

2. Place the corresponding structure file in `benchmarks_with_ground_truth/txt/`
   - Example: `benchmarks_with_ground_truth/txt/mydataset.txt`
   - Format: Each line is `source target` representing an edge

Example TXT file:
```
A B
A C
B D
C D
```

## Running Your First Experiment

### Option 1: Step-by-Step (Recommended for Learning)

```bash
# Step 1: Preprocess your dataset
python scripts/01_preprocess_data.py --dataset mydataset

# Step 2: Train a model (GMM is fast for testing)
python scripts/02_train_model.py --dataset mydataset --model gmm

# Step 3: Generate synthetic data
python scripts/04_generate_synthetic.py --dataset mydataset --model gmm --n-samples 1000

# Step 4: Learn structure from real data
python scripts/05_learn_structure.py --dataset mydataset --algorithm pc --data-type real

# Step 5: Evaluate the model with ground truth structure
python scripts/06_evaluate.py --dataset mydataset --model gmm --structure ground_truth

# Step 6: Evaluate with learned structure
python scripts/06_evaluate.py --dataset mydataset --model gmm --structure pc_real

# Step 7: Compare rankings
python scripts/07_compare_rankings.py --dataset mydataset --structures ground_truth,pc_real --models gmm
```

### Option 2: Automated Batch Processing

Create a shell script `run_experiment.sh`:

```bash
#!/bin/bash

DATASET="mydataset"
MODELS="gmm bayesian_network"
ALGORITHMS="pc ges"

# Preprocess
echo "Preprocessing dataset..."
python scripts/01_preprocess_data.py --dataset $DATASET

# Train all models
for MODEL in $MODELS; do
    echo "Training $MODEL..."
    python scripts/02_train_model.py --dataset $DATASET --model $MODEL
    
    echo "Generating synthetic data..."
    python scripts/04_generate_synthetic.py --dataset $DATASET --model $MODEL --n-samples 1000
done

# Learn structures
for ALGO in $ALGORITHMS; do
    echo "Learning structure with $ALGO..."
    python scripts/05_learn_structure.py --dataset $DATASET --algorithm $ALGO --data-type real
done

# Evaluate all combinations
for MODEL in $MODELS; do
    for STRUCT in ground_truth $ALGORITHMS; do
        echo "Evaluating $MODEL with $STRUCT..."
        python scripts/06_evaluate.py --dataset $DATASET --model $MODEL --structure ${STRUCT}_real || \
        python scripts/06_evaluate.py --dataset $DATASET --model $MODEL --structure $STRUCT
    done
done

# Compare rankings
echo "Comparing rankings..."
STRUCTURES="ground_truth"
for ALGO in $ALGORITHMS; do
    STRUCTURES="$STRUCTURES,${ALGO}_real"
done
python scripts/07_compare_rankings.py --dataset $DATASET --structures $STRUCTURES --models ${MODELS// /,}

echo "Done! Check outputs/ directory for results."
```

Run with:
```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

## Expected Outputs

After running the experiment, you'll find:

```
outputs/
├── models/
│   ├── mydataset_gmm.pkl
│   └── mydataset_bayesian_network.pkl
├── synthetic/
│   ├── mydataset_gmm_synthetic.csv
│   └── mydataset_bayesian_network_synthetic.csv
├── structures/
│   ├── mydataset_pc_real.json
│   ├── mydataset_pc_real.png
│   ├── mydataset_ges_real.json
│   └── mydataset_ges_real.png
├── evaluations/
│   ├── mydataset_gmm_ground_truth_eval.json
│   ├── mydataset_gmm_pc_real_eval.json
│   └── ...
└── rankings/
    └── mydataset_ranking_comparison.json
```

## Interpreting Results

### Evaluation Results

Open `outputs/evaluations/{dataset}_{model}_{structure}_eval.json`:

```json
{
  "quality_score": 0.85,      // Higher is better (0-1)
  "agreement_rate": 0.92,     // CI test agreement (0-1)
  "real_violation_rate": 0.05,
  "synthetic_violation_rate": 0.07
}
```

### Ranking Comparison

Open `outputs/rankings/{dataset}_ranking_comparison.json`:

```json
{
  "rankings": {
    "ground_truth": {
      "order": ["bayesian_network", "gmm"],
      "scores": [0.87, 0.75]
    },
    "pc_real": {
      "order": ["gmm", "bayesian_network"],
      "scores": [0.81, 0.78]
    }
  },
  "comparisons_vs_ground_truth": {
    "pc_real": {
      "kendall_tau": 0.5,           // Rank correlation
      "top_model_match": false       // Did rankings change?
    }
  }
}
```

**Key insight**: If `top_model_match` is `false`, the structure learning algorithm changed which model appears best!

## Common Issues

### Issue: "Dataset not found"
**Solution**: Ensure both CSV and TXT files exist with the same name:
- `benchmarks_with_ground_truth/csv/mydataset.csv`
- `benchmarks_with_ground_truth/txt/mydataset.txt`

### Issue: "Out of memory"
**Solution**: Reduce the number of samples or use a smaller model:
```bash
python scripts/04_generate_synthetic.py --dataset mydataset --model gmm --n-samples 500
```

### Issue: Training takes too long
**Solution**: 
- Use GMM instead of CTGAN for faster testing
- Reduce epochs in `config/models.yaml`

## Next Steps

1. **Try different models**: Train CTGAN and Bayesian Networks
2. **Experiment with structure learning**: Try NOTEARS, FCI, or LiNGAM
3. **Tune hyperparameters**: Use script 03 for better model performance
4. **Compare multiple datasets**: Run on all your datasets
5. **Analyze results**: Use the Jupyter notebook for visualization

## Getting Help

- Check the main README.md for detailed documentation
- Look at example_workflow.ipynb for interactive exploration
- Open a GitHub issue for bugs or questions

## GPU Acceleration

To use GPU (recommended for CTGAN and TabDDPM):

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Models will automatically use GPU if available
python scripts/02_train_model.py --dataset mydataset --model ctgan
```

## Tips for Success

1. **Start small**: Test with one dataset and two models first
2. **Use GMM initially**: It's fast and doesn't require tuning
3. **Save your configs**: Edit `config/*.yaml` for your specific needs
4. **Check outputs**: Always verify results in the outputs/ directory
5. **Compare carefully**: Look at both quality scores and ranking changes

Happy experimenting!
