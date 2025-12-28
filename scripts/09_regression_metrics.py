#!/usr/bin/env python3

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_regression_metrics(real, synth, target_col):
    X_synth = synth.drop(columns=[target_col])
    y_synth = synth[target_col]
    X_real = real.drop(columns=[target_col])
    y_real = real[target_col]

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_synth, y_synth)
    y_pred = reg.predict(X_real)

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    return {'R2': r2, 'MSE': mse, 'MAE': mae}


def main():
    parser = argparse.ArgumentParser(description='Compute regression metrics for synthetic data quality')
    parser.add_argument('--real', type=str, required=True, help='Path to real data CSV')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic data CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--target', type=str, required=False, help='Target column for regression (default: last column)')
    args = parser.parse_args()

    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)

    target_col = args.target if args.target else real.columns[-1]
    if target_col not in real.columns or target_col not in synth.columns:
        raise ValueError(f"Target column '{target_col}' not found in both datasets.")

    metrics = compute_regression_metrics(real, synth, target_col)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved regression metrics to {args.output}")


if __name__ == '__main__':
    main()
