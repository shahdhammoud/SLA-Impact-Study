#!/usr/bin/env python3
import argparse
import os
import json
import matplotlib.pyplot as plt


def load_cautabench_scores(models, eval_dir, dataset):
    scores = {}
    for model in models:
        eval_path = os.path.join(eval_dir, f"{dataset}_{model}_quality.json", "quality_report.json")
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                report = json.load(f)
            scores[model] = 1.0 - report.get('detection_score', 0.0)
        else:
            scores[model] = None
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ranking-json', type=str, required=True)
    parser.add_argument('--eval-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.ranking_json, 'r') as f:
        ranking = json.load(f)

    models = ranking['models']
    algorithms = ranking['algorithms']
    reliability_scores = ranking['reliability_scores']

    cautabench_scores = load_cautabench_scores(models, args.eval_dir, args.dataset)

    for algorithm in algorithms:
        x = []
        y = []
        labels = []
        for model in models:
            rel = reliability_scores.get(algorithm, {}).get(model, None)
            ctb = cautabench_scores.get(model, None)
            if rel is not None and ctb is not None:
                x.append(rel)
                y.append(ctb)
                labels.append(model.upper())
        if not x:
            continue
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, s=200, c=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(x)], edgecolors='black', linewidths=2, alpha=0.8)
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]), fontsize=12, fontweight='bold', xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
        plt.xlabel(f'Reliability Score ({algorithm.upper()}) [%]', fontsize=13, fontweight='bold')
        plt.ylabel('CauTabBench Quality Score', fontsize=13, fontweight='bold')
        plt.title(f'CauTabBench vs Reliability ({algorithm.upper()})', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"{args.dataset}_cautabench_vs_reliability_{algorithm.lower()}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == '__main__':
    main()
