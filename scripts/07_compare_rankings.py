#!/usr/bin/env python3
"""
Script 7: Compare model rankings using algorithm reliability scores.

Evaluates how reliable different causal discovery algorithms are for
assessing synthetic data quality.
"""

import argparse
import os
import sys
import json
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.ranking import RankingComparator
from src.utils.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Compare model rankings using algorithm reliability')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., asia, alarm)')
    parser.add_argument('--algorithms', type=str, default='pc,ges,fci,cdnod',
                       help='Comma-separated list of algorithms (e.g., pc,ges,notears,fci,cdnod)')
    parser.add_argument('--models', type=str, default='gmm,ctgan,tabddpm',
                       help='Comma-separated list of models')
    parser.add_argument('--ground-truth-dir', type=str, default='benchmarks_with_ground_truth/txt',
                       help='Directory containing ground truth txt files')
    parser.add_argument('--structures-dir', type=str, default='outputs/structures',
                       help='Directory containing learned structures')
    parser.add_argument('--eval-dir', type=str, default='outputs/evaluations',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, default='outputs/rankings',
                       help='Output directory for rankings and visualizations')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Generate visualizations')

    args = parser.parse_args()
    logger = setup_logger('compare_rankings', console=True)

    logger.info(f"Comparing rankings for dataset: {args.dataset}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Models: {args.models}")

    algorithms = [a.strip() for a in args.algorithms.split(',')]
    models = [m.strip() for m in args.models.split(',')]

    # Initialize comparator
    comparator = RankingComparator()
    
    # Load ground truths
    logger.info("\n📊 Loading ground truth structures...")
    ground_truth_file = os.path.join(args.ground_truth_dir, f"{args.dataset}.txt")

    if not os.path.exists(ground_truth_file):
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        sys.exit(1)

    for algorithm in algorithms:
        try:
            comparator.load_ground_truth(algorithm, ground_truth_file)
        except Exception as e:
            logger.error(f"Failed to load ground truth for {algorithm}: {e}")
            continue

    # Load real structures (G_real)
    logger.info("\n📊 Loading real data structures...")
    for algorithm in algorithms:
        real_structure_file = os.path.join(args.structures_dir, f"{args.dataset}_{algorithm}_real.json")

        if os.path.exists(real_structure_file):
            try:
                comparator.load_real_structure(algorithm, real_structure_file)
            except Exception as e:
                logger.warning(f"Failed to load real structure {algorithm}: {e}")
        else:
            logger.warning(f"Real structure not found: {real_structure_file}")

    # Load synthetic structures (G_synthetic)
    logger.info("\n📊 Loading synthetic data structures...")
    for model in models:
        for algorithm in algorithms:
            synthetic_structure_file = os.path.join(
                args.structures_dir,
                f"{args.dataset}_{algorithm}_synthetic_{model}.json"
            )

            if os.path.exists(synthetic_structure_file):
                try:
                    comparator.load_synthetic_structure(model, algorithm, synthetic_structure_file)
                except Exception as e:
                    logger.warning(f"Failed to load synthetic structure {model}/{algorithm}: {e}")
            else:
                logger.warning(f"Synthetic structure not found: {synthetic_structure_file}")

    # Load CauTabBench scores (optional)
    logger.info("\n📊 Loading CauTabBench scores...")
    for model in models:
        eval_file = os.path.join(args.eval_dir, f"{args.dataset}_{model}_ground_truth_eval.json")

        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                    cautabench_score = eval_results.get('quality_score', 0.0)
                    comparator.add_cautabench_score(model, cautabench_score)
                    logger.info(f"  {model}: {cautabench_score:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load CauTabBench score for {model}: {e}")

    # Compute all reliability scores
    logger.info("\n📊 Computing algorithm reliability scores...")
    comparator.compute_all_reliability_scores()

    # Display results
    logger.info("\n" + "="*70)
    logger.info("ALGORITHM RELIABILITY SCORES")
    logger.info("="*70)

    for algorithm in algorithms:
        if algorithm in comparator.reliability_scores:
            logger.info(f"\n{algorithm.upper()}:")
            ranking = comparator.rank_models_by_algorithm(algorithm)
            for rank, (model, score) in enumerate(ranking, 1):
                logger.info(f"  {rank}. {model}: {score:.2f}%")

    # Algorithm ranking
    logger.info("\n" + "="*70)
    logger.info("ALGORITHM RELIABILITY RANKING")
    logger.info("="*70)
    algo_ranking = comparator.rank_algorithms_by_reliability()
    for rank, (algorithm, avg_score) in enumerate(algo_ranking, 1):
        logger.info(f"{rank}. {algorithm.upper()}: {avg_score:.2f}% (average)")

    # Overall model ranking
    logger.info("\n" + "="*70)
    logger.info("OVERALL MODEL RANKING (Average across all algorithms)")
    logger.info("="*70)

    summary = comparator.summarize()
    if 'overall_reliability_ranking' in summary['rankings']:
        overall_ranking = summary['rankings']['overall_reliability_ranking']
        for rank, (model, score) in enumerate(zip(overall_ranking['order'], overall_ranking['avg_scores']), 1):
            logger.info(f"{rank}. {model}: {score:.2f}%")

    # CauTabBench ranking
    if comparator.cautabench_scores:
        logger.info("\n" + "="*70)
        logger.info("CAUTABBENCH RANKING")
        logger.info("="*70)
        cautabench_ranking = comparator.rank_by_cautabench()
        for rank, (model, score) in enumerate(cautabench_ranking, 1):
            logger.info(f"{rank}. {model}: {score:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_ranking_comparison.json")
    comparator.save(output_file)
    logger.info(f"\n✅ Results saved to: {output_file}")

    # Generate visualizations
    if args.visualize:
        logger.info("\n📊 Generating visualizations...")
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        comparator.plot_all_visualizations(viz_dir, args.dataset)

    logger.info("\n" + "="*70)
    logger.info("✅ RANKING COMPARISON COMPLETE!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
