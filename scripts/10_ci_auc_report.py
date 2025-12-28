#!/usr/bin/env python3
"""
Script 10: Conditional Independence (CI) ROC AUC Report

Evaluates how well synthetic data preserves the CI structure of the ground-truth graph.
Handles mixed-type data (continuous + categorical).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

import argparse
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from src.evaluation.ci_auc_utils import compute_ci_auc

# You may need to install causallearn for KCI/Fisher's Z
try:
    from causallearn.utils.cit import fisherz, kci
except ImportError:
    fisherz = None
    kci = None
    print("[WARN] causallearn not installed. Please install for full CI test support.")

def identify_variable_types(df, categorical_features=None):
    if categorical_features is not None:
        cat_cols = categorical_features
        num_cols = [col for col in df.columns if col not in cat_cols]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [col for col in df.columns if col not in num_cols]
    return num_cols, cat_cols

def preprocess_for_ci(df, cat_cols):
    if cat_cols:
        return pd.get_dummies(df, columns=cat_cols), True
    return df.copy(), False

def load_ground_truth_graph(graph_path, columns):
    """
    Load ground-truth graph from edge list (txt) and return adjacency matrix matching columns order.
    """
    # Read edge list
    edges = []
    with open(graph_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                src, tgt = parts[0], parts[1]
                edges.append((src, tgt))
    # Build adjacency matrix
    n = len(columns)
    col_idx = {col: i for i, col in enumerate(columns)}
    adj = np.zeros((n, n), dtype=int)
    for src, tgt in edges:
        if src in col_idx and tgt in col_idx:
            adj[col_idx[src], col_idx[tgt]] = 1
    return adj

def generate_ci_queries(adj, columns, processed_columns):
    # Only generate queries for columns that exist in the processed DataFrame
    n = adj.shape[0]
    queries = []
    for i in range(n):
        for j in range(i+1, n):
            X, Y = columns[i], columns[j]
            # For categorical columns, use all one-hot columns
            X_cols = [col for col in processed_columns if col.startswith(f"{X}_") or col == X]
            Y_cols = [col for col in processed_columns if col.startswith(f"{Y}_") or col == Y]
            # Only add queries if both X and Y are present after encoding
            if X_cols and Y_cols:
                queries.append((X_cols[0], Y_cols[0], []))
    return queries

def run_ci_test(df, X, Y, S, num_cols, cat_cols):
    # Use Fisher's Z for all-continuous, KCI for mixed/one-hot
    if callable(fisherz) and all(col in num_cols for col in [X, Y] + S):
        return fisherz(df[X].values, df[Y].values, df[S].values if S else None)[1]
    elif callable(kci):
        return kci(df[X].values, df[Y].values, df[S].values if S else None)[1]
    else:
        # Fallback: Pearson correlation p-value
        from scipy.stats import pearsonr
        return pearsonr(df[X], df[Y])[1]

def get_base_col(col):
    return col.split('_')[0] if '_' in col else col

def main():
    parser = argparse.ArgumentParser(description='CI ROC AUC Report')
    parser.add_argument('--real', type=str, required=True, help='Path to real data CSV')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic data CSV')
    parser.add_argument('--info', type=str, required=True, help='Path to info JSON (for categorical features)')
    parser.add_argument('--graph', type=str, required=True, help='Path to ground-truth adjacency matrix (.txt or .csv)')
    parser.add_argument('--output', type=str, required=True, help='Output directory for report and plots')
    parser.add_argument('--model-path', type=str, default=None, help='Path to best model (optional)')
    args = parser.parse_args()

    # NOTE: For consistency, use the synthetic data file saved from the best Optuna trial:
    # outputs/synthetic/<dataset>_<model>_synthetic_best.csv
    # This guarantees the Y-axis value matches the maximized Optuna metric.

    # Use the best model and test set if available
    synth = pd.read_csv(args.synthetic)
    with open(args.info, 'r') as f:
        info = json.load(f)
    # Compute CI AUC using the shared utility
    auc, y_true, y_score = compute_ci_auc(synth, info, args.graph)
    print(f"ROC AUC (synthetic CI structure vs ground truth): {auc:.4f}")
    # Save report
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'ci_auc_report.json'), 'w') as f:
        json.dump({'roc_auc': auc}, f, indent=2)

    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CI ROC Curve (Synthetic Data)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'ci_roc_curve.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
