import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_spearman_correlation(csv_path, output_path=None, title=None):
    df = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', square=True, cbar_kws={"label": "Spearman Correlation"})
    plt.title(title or f"Spearman Rank Correlation Matrix")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"[Spearman] Heatmap saved to: {output_path}")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Spearman correlation matrix as heatmap.")
    parser.add_argument('--csv', type=str, required=True, help='Path to Spearman correlation CSV')
    parser.add_argument('--output', type=str, default=None, help='Path to save PNG (optional)')
    parser.add_argument('--title', type=str, default=None, help='Plot title (optional)')
    args = parser.parse_args()
    plot_spearman_correlation(args.csv, args.output, args.title)

if __name__ == '__main__':
    main()
