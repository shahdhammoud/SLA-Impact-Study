#!/usr/bin/env python3
"""Script 6: Evaluate generative model using CauTabBench methodology."""

import argparse
import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.cautabbench_eval import CauTabBenchEvaluator
from src.data.loader import DataLoader
from src.structure_learning.base import BaseStructureLearner
from src.utils.logging_utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Evaluate generative model')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--structure', type=str, required=True, help='ground_truth or algorithm_name (e.g., pc_learned)')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed')
    parser.add_argument('--synthetic-dir', type=str, default='outputs/synthetic')
    parser.add_argument('--structure-dir', type=str, default='outputs/structures')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluations')
    parser.add_argument('--method', type=str, default='fisherz', choices=['fisherz', 'chisq', 'gsq'])
    parser.add_argument('--alpha', type=float, default=0.05)
    
    args = parser.parse_args()
    logger = setup_logger('evaluate', console=True)
    logger.info(f"Evaluating {args.model} on {args.dataset} with structure: {args.structure}")
    
    real_data = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_preprocessed.csv"))
    synthetic_data = pd.read_csv(os.path.join(args.synthetic_dir, f"{args.dataset}_{args.model}_synthetic.csv"))
    
    loader = DataLoader()
    if args.structure == 'ground_truth':
        _, causal_graph = loader.load_dataset(args.dataset)
    else:
        learner = BaseStructureLearner(algorithm_name='loaded')
        structure_file = os.path.join(args.structure_dir, f"{args.dataset}_{args.structure}.json")
        learner.load(structure_file)
        causal_graph = learner.get_graph()
    
    logger.info(f"Using structure with {causal_graph.number_of_edges()} edges")
    
    evaluator = CauTabBenchEvaluator(method=args.method, alpha=args.alpha)
    results = evaluator.evaluate(real_data, synthetic_data, causal_graph)
    
    logger.info(f"Quality score: {results['quality_score']:.4f}")
    logger.info(f"Agreement rate: {results['agreement_rate']:.4f}")
    logger.info(f"Real violation rate: {results['real_violation_rate']:.4f}")
    logger.info(f"Synthetic violation rate: {results['synthetic_violation_rate']:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_{args.structure}_eval.json")
    evaluator.save_results(results, output_file)
    logger.info(f"Saved results to {output_file}")

if __name__ == '__main__':
    main()
