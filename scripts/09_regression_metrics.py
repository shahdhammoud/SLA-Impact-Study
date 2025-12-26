#!/usr/bin/env python3
"""
Script 9: Compute regression-based metric (R2) for synthetic data quality

Trains a regressor on synthetic data and tests on real data (and vice versa),
computing R2 for each generative model.
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def compute_regression_r2(real, synth, target_col):
    # Train on synthetic, test on real
    X_synth = synth.drop(columns=[target_col])
    y_synth = synth[target_col]
    X_real = real.drop(columns=[target_col])
    y_real = real[target_col]

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_synth, y_synth)
    y_pred = reg.predict(X_real)

    r2 = r2_score(y_real, y_pred)
    return {'R2': r2}


def main():
    parser = argparse.ArgumentParser(description='Compute regression R2 for synthetic data quality')
    parser.add_argument('--real', type=str, required=True, help='Path to real data CSV')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic data CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--target', type=str, required=False, help='Target column for regression (default: last column)')
    args = parser.parse_args()

    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)

    # Use last column as target if not provided
    target_col = args.target if args.target else real.columns[-1]
    if target_col not in real.columns or target_col not in synth.columns:
        raise ValueError(f"Target column '{target_col}' not found in both datasets.")

    metrics = compute_regression_r2(real, synth, target_col)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved regression R2 to {args.output}")

if __name__ == '__main__':
    main()
