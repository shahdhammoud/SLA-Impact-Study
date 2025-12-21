#!/usr/bin/env python3
"""Script 7: Compare model rankings across different structures."""

import argparse
import os
import sys
import json
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.ranking import RankingComparator
from src.utils.logging_utils import setup_logger
from src.utils.visualization_utils import (
    plot_model_rankings, 
    plot_ranking_comparison,
    plot_consensus_ranking
)

def main():
    parser = argparse.ArgumentParser(description='Compare model rankings')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--structures', type=str, required=True, help='Comma-separated list of structures')
    parser.add_argument('--models', type=str, default='ctgan,gmm,bayesian_network', help='Comma-separated list of models')
    parser.add_argument('--eval-dir', type=str, default='outputs/evaluations')
    parser.add_argument('--output-dir', type=str, default='outputs/rankings')
    parser.add_argument('--baseline', type=str, default='ground_truth')
    
    args = parser.parse_args()
    logger = setup_logger('compare_rankings', console=True)
    logger.info(f"Comparing rankings for {args.dataset}")
    
    structures = args.structures.split(',')
    models = args.models.split(',')
    
    comparator = RankingComparator()
    
    for structure in structures:
        structure = structure.strip()
        rankings = []
        
        for model in models:
            model = model.strip()
            eval_file = os.path.join(args.eval_dir, f"{args.dataset}_{model}_{structure}_eval.json")
            
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    results = json.load(f)
                rankings.append((model, results['quality_score']))
            else:
                logger.warning(f"Evaluation file not found: {eval_file}")
        
        if rankings:
            rankings.sort(key=lambda x: x[1], reverse=True)
            comparator.add_ranking(structure, rankings)
            logger.info(f"Structure {structure}: {[name for name, _ in rankings]}")
    
    if len(comparator.comparisons) < 2:
        logger.error("Need at least 2 structures to compare")
        sys.exit(1)
    
    comparison_results = comparator.compare_rankings(baseline=args.baseline)
    
    logger.info(f"\nComparisons against {args.baseline}:")
    for struct, metrics in comparison_results['comparisons'].items():
        logger.info(f"\n{struct}:")
        logger.info(f"  Kendall's tau: {metrics['kendall_tau']:.3f} (p={metrics['kendall_p_value']:.3f})")
        logger.info(f"  Spearman's rho: {metrics['spearman_rho']:.3f} (p={metrics['spearman_p_value']:.3f})")
        logger.info(f"  Avg positional diff: {metrics['avg_positional_difference']:.2f}")
        logger.info(f"  Top model match: {metrics['top_model_match']}")
    
    consensus = comparator.get_consensus_ranking()
    logger.info(f"\nConsensus ranking: {[name for name, _ in consensus]}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_ranking_comparison.json")
    comparator.save(output_file)
    logger.info(f"\nSaved results to {output_file}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # Plot overall model rankings across all structures
    viz_file = os.path.join(args.output_dir, f"{args.dataset}_rankings_overview.png")
    plot_model_rankings(
        rankings_dict=comparator.comparisons,
        save_path=viz_file,
        title=f"Model Rankings Across Structure Learning Algorithms - {args.dataset}"
    )
    
    # Plot comparison with baseline if baseline exists
    if args.baseline in comparator.comparisons:
        baseline_ranking = comparator.comparisons[args.baseline]
        
        for struct_name, struct_ranking in comparator.comparisons.items():
            if struct_name != args.baseline:
                compare_viz_file = os.path.join(
                    args.output_dir, 
                    f"{args.dataset}_ranking_{args.baseline}_vs_{struct_name}.png"
                )
                plot_ranking_comparison(
                    baseline_ranking=baseline_ranking,
                    comparison_ranking=struct_ranking,
                    baseline_name=args.baseline,
                    comparison_name=struct_name,
                    save_path=compare_viz_file
                )
    
    # Plot consensus ranking
    consensus_viz_file = os.path.join(args.output_dir, f"{args.dataset}_consensus_ranking.png")
    plot_consensus_ranking(
        rankings_dict=comparator.comparisons,
        consensus_ranking=consensus,
        save_path=consensus_viz_file
    )
    
    logger.info("\nAll visualizations saved successfully!")

if __name__ == '__main__':
    main()
