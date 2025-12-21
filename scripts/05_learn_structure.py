#!/usr/bin/env python3
"""Script 5: Learn causal structure from data."""

import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.structure_learning.pc import PCLearner
from src.structure_learning.ges import GESLearner
from src.structure_learning.notears import NOTEARSLearner
from src.structure_learning.fci import FCILearner
from src.structure_learning.lingam import LiNGAMLearner
from src.data.loader import DataLoader
from src.utils.graph_utils import compute_graph_metrics, save_graph_visualization
from src.utils.logging_utils import setup_logger

LEARNERS = {'pc': PCLearner, 'ges': GESLearner, 'notears': NOTEARSLearner, 'fci': FCILearner, 'lingam': LiNGAMLearner}

def main():
    parser = argparse.ArgumentParser(description='Learn causal structure')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True, choices=['pc', 'ges', 'notears', 'fci', 'lingam'])
    parser.add_argument('--data-type', type=str, required=True, choices=['real', 'synthetic'])
    parser.add_argument('--model', type=str, default=None, help='Model name if data_type is synthetic')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed')
    parser.add_argument('--synthetic-dir', type=str, default='outputs/synthetic')
    parser.add_argument('--output-dir', type=str, default='outputs/structures')
    
    args = parser.parse_args()
    logger = setup_logger('structure_learning', console=True)
    logger.info(f"Learning structure with {args.algorithm} from {args.data_type} data")
    
    if args.data_type == 'real':
        data_file = os.path.join(args.data_dir, f"{args.dataset}_preprocessed.csv")
        suffix = 'real'
    else:
        if not args.model:
            logger.error("--model is required when data_type is 'synthetic'")
            sys.exit(1)
        data_file = os.path.join(args.synthetic_dir, f"{args.dataset}_{args.model}_synthetic.csv")
        suffix = f"synthetic_{args.model}"
    
    data = pd.read_csv(data_file)
    logger.info(f"Loaded data: {len(data)} samples, {len(data.columns)} features")
    
    learner = LEARNERS[args.algorithm]()
    learned_graph = learner.fit(data)
    logger.info(f"Learned graph: {learned_graph.number_of_nodes()} nodes, {learned_graph.number_of_edges()} edges")
    
    loader = DataLoader()
    _, true_graph = loader.load_dataset(args.dataset)
    
    metrics = compute_graph_metrics(true_graph, learned_graph)
    logger.info(f"Metrics: SHD={metrics['shd']:.2f}% (absolute: {metrics['shd_absolute']}), F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    structure_file = os.path.join(args.output_dir, f"{args.dataset}_{args.algorithm}_{suffix}.json")
    learner.save(structure_file)
    logger.info(f"Saved structure to {structure_file}")
    
    viz_file = os.path.join(args.output_dir, f"{args.dataset}_{args.algorithm}_{suffix}.png")
    save_graph_visualization(learned_graph, viz_file, title=f"{args.algorithm.upper()} - {args.dataset} ({suffix})")
    logger.info(f"Saved visualization to {viz_file}")

if __name__ == '__main__':
    main()
