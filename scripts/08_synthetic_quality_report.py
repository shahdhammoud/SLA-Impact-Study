#!/usr/bin/env python3
"""
Script 8: Synthetic Data Quality Report

Compares real and synthetic data using a suite of statistical and ML-based tests.
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def marginal_tests(real, synth, categorical):
    results = {}
    for col in real.columns:
        if col in categorical:
            real_counts = real[col].value_counts(normalize=True)
            synth_counts = synth[col].value_counts(normalize=True)
            all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
            real_vec = [real_counts.get(cat, 0) for cat in all_cats]
            synth_vec = [synth_counts.get(cat, 0) for cat in all_cats]
            chi2, p, _, _ = chi2_contingency([real_vec, synth_vec])
            results[col] = {'test': 'chi2', 'p_value': float(p)}
        else:
            stat, p = ks_2samp(real[col], synth[col])
            results[col] = {'test': 'ks', 'p_value': float(p)}
    return results

def correlation_matrix(df):
    # Use Pearson for continuous, Cramér's V for categorical
    return df.corr(method='pearson').fillna(0)

def correlation_similarity(real, synth):
    real_corr = correlation_matrix(real)
    synth_corr = correlation_matrix(synth)
    diff = np.linalg.norm(real_corr.values - synth_corr.values)
    return float(diff)

def coverage(real, synth):
    missing = {}
    for col in real.columns:
        real_vals = set(real[col].unique())
        synth_vals = set(synth[col].unique())
        missing_vals = real_vals - synth_vals
        if missing_vals:
            missing[col] = list(missing_vals)
    return missing

def basic_stats(real, synth):
    stats = {}
    for col in real.columns:
        stats[col] = {
            'real_mean': float(np.mean(real[col])),
            'synth_mean': float(np.mean(synth[col])),
            'real_std': float(np.std(real[col])),
            'synth_std': float(np.std(synth[col]))
        }
    return stats

def detection_score(real, synth):
    # Label and shuffle
    real['__label__'] = 0
    synth['__label__'] = 1
    df = pd.concat([real, synth], axis=0).sample(frac=1, random_state=42)
    X = df.drop(columns='__label__')
    y = df['__label__']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return float(acc)

def uniqueness(synth):
    n_unique = len(synth.drop_duplicates())
    n_total = len(synth)
    return float(n_unique) / n_total

def plot_histograms(real, synth, outdir):
    os.makedirs(outdir, exist_ok=True)
    for col in real.columns:
        plt.figure()
        plt.hist(real[col], bins=20, alpha=0.5, label='Real')
        plt.hist(synth[col], bins=20, alpha=0.5, label='Synthetic')
        plt.title(f"{col} - Real vs Synthetic")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Synthetic Data Quality Report')
    parser.add_argument('--real', type=str, required=True, help='Path to real data CSV')
    parser.add_argument('--synthetic', type=str, required=True, help='Path to synthetic data CSV')
    parser.add_argument('--info', type=str, required=True, help='Path to info JSON (for categorical features)')
    parser.add_argument('--output', type=str, required=True, help='Output directory for report and plots')
    args = parser.parse_args()

    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)
    with open(args.info, 'r') as f:
        info = json.load(f)
    categorical = info.get('categorical_features', [])

    report = {}
    report['marginal_tests'] = marginal_tests(real, synth, categorical)
    report['correlation_similarity'] = correlation_similarity(real, synth)
    report['coverage'] = coverage(real, synth)
    report['basic_stats'] = basic_stats(real, synth)
    report['detection_score'] = detection_score(real.copy(), synth.copy())
    report['uniqueness'] = uniqueness(synth)

    # Save report
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'quality_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Plot histograms
    plot_histograms(real, synth, args.output)

    # Print warnings if quality is low
    warn = False
    for col, res in report['marginal_tests'].items():
        if res['p_value'] < 0.01:
            print(f"[WARN] Marginal distribution for {col} differs significantly (p={res['p_value']:.3g})")
            warn = True
    if report['detection_score'] > 0.7:
        print(f"[WARN] Detection score is high ({report['detection_score']:.2f}) - synthetic data is easily distinguished from real.")
        warn = True
    if report['uniqueness'] < 0.8:
        print(f"[WARN] Synthetic data has low uniqueness ({report['uniqueness']:.2f})")
        warn = True
    if report['coverage']:
        print(f"[WARN] Some real categories are missing in synthetic data: {report['coverage']}")
        warn = True
    if not warn:
        print("Synthetic data quality is acceptable.")

if __name__ == '__main__':
    main()

