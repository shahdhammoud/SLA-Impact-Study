#!/usr/bin/env python3
"""Script 3: Tune model hyperparameters using Optuna."""

import argparse
import os
import sys
import yaml
import pandas as pd
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ctgan_wrapper import CTGANWrapper
from src.models.gmm_wrapper import GMMWrapper
from src.models.bayesian_network import BayesianNetworkWrapper
from src.tuning.optuna_tuner import OptunaTuner, create_param_space_from_config
from src.data.loader import DataLoader
from src.utils.logging_utils import setup_logger

MODEL_CLASSES = {'ctgan': CTGANWrapper, 'gmm': GMMWrapper, 'bayesian_network': BayesianNetworkWrapper}

def main():
    parser = argparse.ArgumentParser(description='Tune model hyperparameters')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['ctgan', 'gmm', 'bayesian_network'])
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--config', type=str, default='config/models.yaml')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed')
    parser.add_argument('--output-dir', type=str, default='outputs/models')
    
    args = parser.parse_args()
    logger = setup_logger('tune', console=True)
    logger.info(f"Tuning {args.model} on {args.dataset} with {args.trials} trials")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['models'][args.model]
    param_space = create_param_space_from_config(model_config)
    
    data = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_preprocessed.csv"))
    train_size = int(0.8 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    
    with open(os.path.join('data/info', f"{args.dataset}_info.json"), 'r') as f:
        feature_info = json.load(f)
    
    loader = DataLoader()
    _, structure = loader.load_dataset(args.dataset)
    
    tuner = OptunaTuner(MODEL_CLASSES[args.model], args.model, n_trials=args.trials)
    results = tuner.tune(train_data, val_data, structure, param_space, 
                        categorical_columns=feature_info['categorical_features'] if args.model == 'ctgan' else None)
    
    os.makedirs(args.output_dir, exist_ok=True)
    study_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_study.pkl")
    tuner.save_study(study_file)
    
    logger.info(f"Best params: {results['best_params']}")
    logger.info(f"Best score: {results['best_value']}")
    logger.info("Tuning complete!")

if __name__ == '__main__':
    main()
