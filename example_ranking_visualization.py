#!/usr/bin/env python3
"""
Example script demonstrating the complete ranking and visualization workflow.

This script shows how to:
1. Load ground truth, real, and synthetic structures
2. Compute algorithm reliability scores
3. Generate comprehensive visualizations
4. Save results
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.evaluation.ranking import RankingComparator

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        Algorithm Reliability & Model Ranking Demonstration          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    dataset = "asia"
    algorithms = ["pc"]  # Add more: ["pc", "ges", "fci"]
    models = ["gmm", "ctgan", "tabddpm"]

    # Initialize comparator
    comparator = RankingComparator()

    print("\n📂 Step 1: Loading ground truth structures...")
    print("-" * 70)

    # Load ground truth (same file for all algorithms)
    ground_truth_file = f"benchmarks_with_ground_truth/txt/{dataset}.txt"
    for algorithm in algorithms:
        try:
            comparator.load_ground_truth(algorithm, ground_truth_file)
        except FileNotFoundError:
            print(f"⚠️  Ground truth file not found: {ground_truth_file}")
            continue

    print("\n📂 Step 2: Loading real data structures (G_real)...")
    print("-" * 70)

    for algorithm in algorithms:
        real_structure = f"outputs/structures/{dataset}_{algorithm}_real.json"
        if os.path.exists(real_structure):
            comparator.load_real_structure(algorithm, real_structure)
        else:
            print(f"⚠️  Real structure not found: {real_structure}")

    print("\n📂 Step 3: Loading synthetic data structures (G_synthetic)...")
    print("-" * 70)

    for model in models:
        for algorithm in algorithms:
            synthetic_structure = f"outputs/structures/{dataset}_{algorithm}_synthetic_{model}.json"
            if os.path.exists(synthetic_structure):
                comparator.load_synthetic_structure(model, algorithm, synthetic_structure)
            else:
                print(f"⚠️  Synthetic structure not found: {synthetic_structure}")

    print("\n📂 Step 4: Loading CauTabBench scores (optional)...")
    print("-" * 70)

    # Example CauTabBench scores (replace with actual loaded scores)
    cautabench_scores = {
        'gmm': 0.96,
        'ctgan': 0.85,
        'tabddpm': 0.72
    }

    for model, score in cautabench_scores.items():
        comparator.add_cautabench_score(model, score)
        print(f"  {model}: {score:.4f}")

    print("\n📊 Step 5: Computing algorithm reliability scores...")
    print("-" * 70)

    comparator.compute_all_reliability_scores()

    # Display results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Model rankings by algorithm
    for algorithm in algorithms:
        ranking = comparator.rank_models_by_algorithm(algorithm)
        if ranking:
            print(f"\n{algorithm.upper()} - Model Ranking:")
            for rank, (model, score) in enumerate(ranking, 1):
                print(f"  {rank}. {model}: {score:.2f}%")

    # Algorithm reliability ranking
    print("\nAlgorithm Reliability Ranking:")
    algo_ranking = comparator.rank_algorithms_by_reliability()
    for rank, (algorithm, score) in enumerate(algo_ranking, 1):
        print(f"  {rank}. {algorithm}: {score:.2f}%")

    # Overall model ranking
    summary = comparator.summarize()
    if 'overall_reliability_ranking' in summary['rankings']:
        print("\nOverall Model Ranking (Average across algorithms):")
        overall = summary['rankings']['overall_reliability_ranking']
        for rank, (model, score) in enumerate(zip(overall['order'], overall['avg_scores']), 1):
            print(f"  {rank}. {model}: {score:.2f}%")

    # CauTabBench ranking
    print("\nCauTabBench Ranking:")
    cautabench_ranking = comparator.rank_by_cautabench()
    for rank, (model, score) in enumerate(cautabench_ranking, 1):
        print(f"  {rank}. {model}: {score:.4f}")

    print("\n" + "="*70)
    print("📊 Step 6: Generating visualizations...")
    print("="*70)

    # Save JSON results
    os.makedirs('outputs/rankings', exist_ok=True)
    json_file = f'outputs/rankings/{dataset}_ranking_comparison.json'
    comparator.save(json_file)

    # Generate all visualizations
    viz_dir = f'outputs/rankings/visualizations'
    comparator.plot_all_visualizations(viz_dir, dataset)

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  📄 JSON Results: {json_file}")
    print(f"  📊 Visualizations: {viz_dir}/")
    print(f"     - {dataset}_reliability_heatmap.png")
    print(f"     - {dataset}_algorithm_comparison.png")
    print(f"     - {dataset}_model_rankings.png")
    print(f"     - {dataset}_cautabench_vs_reliability.png")
    print("\n🎉 All done! Check the output files for results.\n")

if __name__ == "__main__":
    main()

