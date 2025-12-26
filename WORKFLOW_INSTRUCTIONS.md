# Workflow Instructions for Synthetic Data Quality and Causal Structure Evaluation

This guide explains how to run the full workflow for evaluating generative models and causal discovery algorithms on any tabular dataset. Follow these steps for reproducible results and interpretable outputs.

---

## 1. Preprocess the Dataset
- Prepare your dataset in CSV format.
- Run the preprocessing script to clean and format the data.

## 2. Tune Generative Models
- Use Optuna or similar for hyperparameter tuning.
- Example: `python scripts/03_tune_model.py --dataset <name> --model <model> --trials <N>`

## 3. Train Generative Models
- Train each model using the best parameters found in tuning.
- Example: `python scripts/02_train_model.py --dataset <name> --model <model> --use-best-params`

## 4. Generate Synthetic Data
- Use the trained models to generate synthetic datasets.
- Example: `python scripts/04_generate_synthetic.py --dataset <name> --model <model>`

## 5. Learn Causal Structures
- Apply each causal discovery algorithm to both real and synthetic data.
- Example: `python scripts/05_learn_structure.py --dataset <name> --algorithm <alg> --data-type real`
- Example: `python scripts/05_learn_structure.py --dataset <name> --algorithm <alg> --data-type synthetic --model <model>`

## 6. Compute Graph Recovery Metrics
- For each (model, algorithm) pair, compute:
  - SHD (Structural Hamming Distance)
  - Precision
  - Recall
  - F1 Score
  - Composite Score (average of normalized F1, Precision, Recall)

## 7. Calculate Reliability Scores
- For each (model, algorithm) pair, compute reliability score:
  - `score = shd_synthetic_real / shd_synthetic_true`
  - If `shd_synthetic_true == 0`, set score to 1.0

## 8. Generate Plots
- For each algorithm, generate plots:
  - X-axis: Reliability score
  - Y-axis: F1 Score, SHD, Precision, Recall, Composite Score (one plot per metric)
  - Normalize axes for interpretability

## 9. Rank Models and Algorithms
- Rank generative models by each metric for each algorithm
- Compute Spearman’s rank correlation between rankings

## 10. Save Results
- Save all metrics, rankings, and plots to the outputs directory

## 11. Review and Interpret Results
- Use the saved tables and plots to compare model and algorithm performance
- Refer to the explanations in the code and this file for metric definitions

## 12. CI ROC AUC Evaluation (New)
- After obtaining synthetic data, evaluate the quality of causal inference using the CI ROC AUC metric.
- This step is crucial for understanding how well the synthetic data can reproduce the conditional independence structure of the real data.

### Step-by-step Instructions

1. **Preprocess Data**
   - Prepare real and synthetic datasets as CSV files.
   - Ensure you have the info JSON (with categorical features) and the ground-truth adjacency matrix (txt or csv).

2. **Run CI ROC AUC Evaluation**
   - Use the following command:

     ```bash
     python scripts/10_ci_auc_report.py \
       --real data/preprocessed/<DATASET>_preprocessed.csv \
       --synthetic outputs/synthetic/<DATASET>_<MODEL>_synthetic.csv \
       --info data/info/<DATASET>_info.json \
       --graph benchmarks_with_ground_truth/txt/<DATASET>.txt \
       --output outputs/evaluations/<DATASET>_<MODEL>_ci_auc
     ```
   - Replace `<DATASET>` and `<MODEL>` with your dataset/model names.

3. **Outputs**
   - `ci_auc_report.json`: Contains the ROC AUC value (how well synthetic data preserves CI structure).
   - `ci_roc_curve.png`: ROC curve plot for visual inspection.

4. **Visualization**
   - Use the ROC AUC value as the Y-axis in your model comparison plots.
   - Update your visualization scripts to support this metric if not already present.

5. **Batch Processing**
   - Repeat for each model/dataset as needed.

---

## Notes
- All code is documented with clear explanations for each metric and function.
- No emojis are used in code or output.
- This workflow is reusable for any tabular dataset. Just update the dataset name and repeat the steps.
- The script for CI ROC AUC evaluation handles mixed-type data (continuous + categorical).
- It requires `causallearn` for advanced CI tests (Fisher's Z, KCI). Falls back to Pearson correlation if not available.
- For best results, ensure your info JSON correctly lists all categorical features.
- The ROC AUC metric summarizes how well the synthetic data preserves the conditional independence structure of the ground-truth graph.

---

For questions or troubleshooting, refer to the code docstrings or contact the workflow maintainer.
