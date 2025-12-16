#!/usr/bin/env python3
"""Script 4: Generate synthetic data from trained model."""

import argparse
import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ctgan_wrapper import CTGANWrapper
from src.models.gmm_wrapper import GMMWrapper
from src.models.bayesian_network import BayesianNetworkWrapper
from src.data.preprocessor import DataPreprocessor
from src.utils.logging_utils import setup_logger

MODEL_CLASSES = {'ctgan': CTGANWrapper, 'gmm': GMMWrapper, 'bayesian_network': BayesianNetworkWrapper}

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['ctgan', 'gmm', 'bayesian_network'])
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--model-dir', type=str, default='outputs/models')
    parser.add_argument('--output-dir', type=str, default='outputs/synthetic')
    parser.add_argument('--denormalize', action='store_true', default=True)
    
    args = parser.parse_args()
    logger = setup_logger('generate', console=True)
    logger.info(f"Generating {args.n_samples} samples from {args.model} trained on {args.dataset}")
    
    model_file = os.path.join(args.model_dir, f"{args.dataset}_{args.model}.pkl")
    model = MODEL_CLASSES[args.model](model_name=args.model)
    model.load(model_file)
    logger.info(f"Loaded model from {model_file}")
    
    synthetic_data = model.sample(args.n_samples)
    logger.info(f"Generated {len(synthetic_data)} samples")
    
    if args.denormalize:
        preprocessor = DataPreprocessor()
        preprocessor.load_metadata(os.path.join('data/info', f"{args.dataset}_info.json"))
        synthetic_data = preprocessor.inverse_transform(synthetic_data, denormalize=True)
        logger.info("Denormalized synthetic data")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_synthetic.csv")
    synthetic_data.to_csv(output_file, index=False)
    logger.info(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
