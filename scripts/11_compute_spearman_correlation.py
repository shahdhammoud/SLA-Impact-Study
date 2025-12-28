import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

def load_ranking_json(json_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_spearman_matrix(model_scores):
    models = list(model_scores.keys())
    scores = np.array([model_scores[m] for m in models])
    n = len(models)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j], _ = spearmanr(scores[i], scores[j]) if i != j else (1.0, None)
    return pd.DataFrame(matrix, index=models, columns=models)

def main():
    parser = argparse.ArgumentParser(description="Compute Spearman rank correlation matrix for model rankings.")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output', required=True, help='Output CSV file')
    args = parser.parse_args()

    ranking_json = f"outputs/rankings/{args.dataset}_ranking_comparison.json"
    if not os.path.exists(ranking_json):
        print(f"Ranking file not found: {ranking_json}")
        return
    data = load_ranking_json(ranking_json)
    reliability_scores = data.get('reliability_scores', {})
    models = data.get('models', [])
    algorithms = data.get('algorithms', [])

    score_matrix = []
    for model in models:
        row = []
        for algo in algorithms:
            row.append(reliability_scores.get(algo, {}).get(model, np.nan))
        score_matrix.append(row)
    score_matrix = np.array(score_matrix)
    n = len(models)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j], _ = spearmanr(score_matrix[i], score_matrix[j]) if i != j else (1.0, None)
    df = pd.DataFrame(matrix, index=models, columns=models)
    df.to_csv(args.output)
    print(f"[Spearman] All algorithms: correlation matrix saved to: {args.output}")

if __name__ == "__main__":
    main()
