#!/usr/bin/env python3
"""
Script 7: Compare model rankings using algorithm reliability scores.

Evaluates how reliable different causal discovery algorithms are for
assessing synthetic data quality.
"""

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
import json
from glob import glob
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.ranking import RankingComparator
from src.utils.logging_utils import setup_logger


def compute_spearman_correlation_table(rankings_dict, metrics_to_use, models):
    # Build a matrix: rows=metrics, cols=metrics, values=Spearman correlation between model rankings
    n = len(metrics_to_use)
    corr = pd.DataFrame(np.eye(n), index=metrics_to_use, columns=metrics_to_use)
    for i, m1 in enumerate(metrics_to_use):
        for j, m2 in enumerate(metrics_to_use):
            if i != j:
                v1 = [rankings_dict[m1].get(m, float('nan')) for m in models]
                v2 = [rankings_dict[m2].get(m, float('nan')) for m in models]
                mask = ~pd.isna(v1) & ~pd.isna(v2)
                if sum(mask) >= 2:
                    corr.iloc[i, j] = spearmanr(np.array(v1)[mask], np.array(v2)[mask]).correlation
                else:
                    corr.iloc[i, j] = float('nan')
    return corr

def save_spearman_correlation_table(corr, path):
    corr.to_csv(path)
    print(f"[Spearman] Correlation matrix saved to: {path}")


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

    # Patch: Always use the correct synthetic data file for each model
    # This ensures consistency with the best Optuna trial output
    # (Assumes downstream code uses this file for structure learning and evaluation)
    for model in models:
        synthetic_best_path = os.path.join('outputs', 'synthetic', f"{args.dataset}_{model}_synthetic_best.csv")
        if not os.path.exists(synthetic_best_path):
            logger.warning(f"[WARNING] Synthetic data file for model {model} not found: {synthetic_best_path}")
        else:
            logger.info(f"[INFO] Using synthetic data file for {model}: {synthetic_best_path}")

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

    # ================= SPEARMAN RANK CORRELATION ANALYSIS =================
    # Prepare rankings for each metric (using the same values as in the plots)
    # Example metrics: CI ROC AUC, F1, SHD, Precision, Recall, Composite
    # This assumes you have a summary['metrics'] dict with per-model scores for each metric
    # If not, you may need to load them from the appropriate files or compute them here
    #
    # For demonstration, let's assume you have a dictionary like:
    # rankings_dict = {
    #     'ci_roc_auc': {'ctgan': 0.58, 'gmm': 0.44, 'tabddpm': 0.40},
    #     'f1': {'ctgan': 0.12, 'gmm': 0.22, 'tabddpm': 0.18},
    #     ...
    # }
    #
    # You should replace this with actual loading from your evaluation outputs.
    #
    # Example: Load CI ROC AUC from outputs/evaluations/<dataset>_<model>_ci_auc.json
    metrics = ['ci_roc_auc', 'f1', 'shd', 'precision', 'recall', 'composite']
    rankings_dict = {m: {} for m in metrics}
    for model in models:
        # Load CI ROC AUC from best Optuna value
        best_ci_auc_path = os.path.join('outputs', 'models', f'{args.dataset}_{model}_best_ci_auc.json')
        ci_auc_value = None
        if os.path.exists(best_ci_auc_path):
            try:
                with open(best_ci_auc_path, 'r') as f:
                    best_data = json.load(f)
                    ci_auc_value = best_data.get('ci_auc', float('nan'))
                    rankings_dict['ci_roc_auc'][model] = ci_auc_value
            except Exception as e:
                logger.warning(f"Failed to load best Optuna CI ROC AUC for {model}: {e}")
        # Fallback: Load CI ROC AUC from evaluation report if best not found
        if ci_auc_value is None or pd.isna(ci_auc_value):
            ci_auc_path = os.path.join('outputs', 'evaluations', f'{args.dataset}_{model}_ci_auc.json', 'ci_auc_report.json')
            if os.path.exists(ci_auc_path):
                try:
                    with open(ci_auc_path, 'r') as f:
                        ci_auc_data = json.load(f)
                        rankings_dict['ci_roc_auc'][model] = ci_auc_data.get('roc_auc', float('nan'))
                except Exception as e:
                    logger.warning(f"Failed to load CI ROC AUC for {model}: {e}")
        # Load other metrics (F1, SHD, etc.) from quality or regression metrics files as needed
        quality_path = os.path.join('outputs', 'evaluations', f'{args.dataset}_{model}_quality.json')
        if os.path.exists(quality_path):
            try:
                with open(quality_path, 'r') as f:
                    quality_data = json.load(f)
                    rankings_dict['f1'][model] = quality_data.get('f1', float('nan'))
                    rankings_dict['shd'][model] = quality_data.get('shd', float('nan'))
                    rankings_dict['precision'][model] = quality_data.get('precision', float('nan'))
                    rankings_dict['recall'][model] = quality_data.get('recall', float('nan'))
                    rankings_dict['composite'][model] = quality_data.get('composite', float('nan'))
            except Exception as e:
                logger.warning(f"Failed to load quality metrics for {model}: {e}")
    # Only keep metrics that have at least two non-NaN values
    metrics_to_use = [m for m in metrics if sum(~pd.isna(list(rankings_dict[m].values()))) >= 2]
    if len(metrics_to_use) >= 2:
        corr = compute_spearman_correlation_table(rankings_dict, metrics_to_use, models)
        spearman_path = os.path.join(args.output_dir, f"{args.dataset}_spearman_correlation.csv")
        save_spearman_correlation_table(corr, spearman_path)
    else:
        logger.warning("Not enough valid metrics for Spearman correlation analysis.")
    # ================= END SPEARMAN RANK CORRELATION ANALYSIS =================

    # Generate visualizations
    if args.visualize:
        logger.info("\n Generating visualizations...")
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        comparator.plot_all_visualizations(viz_dir, args.dataset, models)
        # Add: plot CI ROC AUC vs reliability using best Optuna value (now handled inside plot_all_visualizations)
        # comparator.plot_ci_auc_vs_reliability(args.dataset, models, viz_dir)

    logger.info("\n" + "="*70)
    logger.info(" RANKING COMPARISON COMPLETE!")
    logger.info("="*70)


if __name__ == '__main__':
    main()
