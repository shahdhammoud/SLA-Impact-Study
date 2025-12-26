#!/usr/bin/env python3
"""
Script 10: Conditional Independence (CI) ROC AUC Report

Evaluates how well synthetic data preserves the CI structure of the ground-truth graph.
Handles mixed-type data (continuous + categorical).
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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
    args = parser.parse_args()

    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)
    with open(args.info, 'r') as f:
        info = json.load(f)
    categorical = info.get('categorical_features', [])

    # Identify variable types
    num_cols, cat_cols = identify_variable_types(real, categorical)
    # Preprocess (one-hot for categorical)
    real_proc, _ = preprocess_for_ci(real, cat_cols)
    synth_proc, _ = preprocess_for_ci(synth, cat_cols)

    columns = real.columns.tolist()
    processed_columns = synth_proc.columns.tolist()
    # Load ground-truth graph as adjacency matrix
    adj = load_ground_truth_graph(args.graph, columns)
    queries = generate_ci_queries(adj, columns, processed_columns)

    # For each query, run CI test on real and synthetic
    y_true = []  # 1 if d-separated (independent), 0 if d-connected (dependent)
    y_score = [] # p-value from CI test on synthetic
    for X, Y, S in queries:
        # Map one-hot column names back to base column names for adjacency lookup
        X_base = get_base_col(X)
        Y_base = get_base_col(Y)
        i, j = columns.index(X_base), columns.index(Y_base)
        dsep = int(adj[i, j] == 0 and adj[j, i] == 0)
        y_true.append(dsep)
        # Use synthetic data for CI test
        pval = run_ci_test(synth_proc, X, Y, S, num_cols, cat_cols)
        y_score.append(pval)

    # Compute ROC AUC
    auc = roc_auc_score(y_true, y_score)
    print(f"ROC AUC (synthetic CI structure vs ground truth): {auc:.4f}")

    # Save report
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'ci_auc_report.json'), 'w') as f:
        json.dump({'roc_auc': auc}, f, indent=2)

    # Plot ROC curve
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
