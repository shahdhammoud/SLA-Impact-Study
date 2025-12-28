#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.evaluation.ranking import RankingComparator

def main():
    dataset = "asia"
    algorithms = ["pc"]  # Add more: ["pc", "ges", "fci"]
    models = ["gmm", "ctgan", "tabddpm"]

    comparator = RankingComparator()

    ground_truth_file = f"benchmarks_with_ground_truth/txt/{dataset}.txt"
    for algorithm in algorithms:
        try:
            comparator.load_ground_truth(algorithm, ground_truth_file)
        except FileNotFoundError:
            continue

    for algorithm in algorithms:
        real_structure = f"outputs/structures/{dataset}_{algorithm}_real.json"
        if os.path.exists(real_structure):
            comparator.load_real_structure(algorithm, real_structure)

    for model in models:
        for algorithm in algorithms:
            synthetic_structure = f"outputs/structures/{dataset}_{algorithm}_synthetic_{model}.json"
            if os.path.exists(synthetic_structure):
                comparator.load_synthetic_structure(model, algorithm, synthetic_structure)

    cautabench_scores = {
        'gmm': 0.96,
        'ctgan': 0.85,
        'tabddpm': 0.72
    }

    for model, score in cautabench_scores.items():
        comparator.add_cautabench_score(model, score)

    comparator.compute_all_reliability_scores()

    for algorithm in algorithms:
        ranking = comparator.rank_models_by_algorithm(algorithm)
        if ranking:
            for rank, (model, score) in enumerate(ranking, 1):
                print(f"  {rank}. {model}: {score:.2f}%")

    algo_ranking = comparator.rank_algorithms_by_reliability()
    for rank, (algorithm, score) in enumerate(algo_ranking, 1):
        print(f"  {rank}. {algorithm}: {score:.2f}%")

    summary = comparator.summarize()
    if 'overall_reliability_ranking' in summary['rankings']:
        overall = summary['rankings']['overall_reliability_ranking']
        for rank, (model, score) in enumerate(zip(overall['order'], overall['avg_scores']), 1):
            print(f"  {rank}. {model}: {score:.2f}%")

    cautabench_ranking = comparator.rank_by_cautabench()
    for rank, (model, score) in enumerate(cautabench_ranking, 1):
        print(f"  {rank}. {model}: {score:.4f}")

    os.makedirs('outputs/rankings', exist_ok=True)
    json_file = f'outputs/rankings/{dataset}_ranking_comparison.json'
    comparator.save(json_file)

    viz_dir = f'outputs/rankings/visualizations'
    comparator.plot_all_visualizations(viz_dir, dataset)

if __name__ == "__main__":
    main()
