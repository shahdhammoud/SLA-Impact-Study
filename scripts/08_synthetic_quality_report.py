#!/usr/bin/env python3
import os
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
import sys
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Quality Report")
    parser.add_argument('--real', type=str, required=True, help='Path to real data CSV')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic data CSV')
    parser.add_argument('--info', type=str, required=True, help='Path to info JSON')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    args = parser.parse_args()
    real = pd.read_csv(args.real)
    synthetic = pd.read_csv(args.synthetic)
    with open(args.info, 'r') as f:
        info = json.load(f)
    target = info.get('target')
    y_true = real[target].values
    y_pred = synthetic[target].values
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    report = {
        'r2': r2,
        'mse': mse,
        'mae': mae
    }
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Quality report saved to: {args.output}")

if __name__ == '__main__':
    main()
