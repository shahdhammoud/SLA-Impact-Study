#!/usr/bin/env python3
"""Script 3: Tune model hyperparameters using Optuna."""

import argparse
import os
import sys
import yaml
import pandas as pd
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ctgan_wrapper import CTGANWrapper
from src.models.gmm_wrapper import GMMWrapper
from src.models.bayesian_network import BayesianNetworkWrapper
from src.models.tabddpm_wrapper import TabDDPMWrapper
from src.tuning.optuna_tuner import OptunaTuner, create_param_space_from_config
from src.data.loader import DataLoader
from src.utils.logging_utils import setup_logger
from src.evaluation.ci_auc_utils import compute_ci_auc

MODEL_CLASSES = {'ctgan': CTGANWrapper, 'gmm': GMMWrapper, 'bayesian_network': BayesianNetworkWrapper, 'tabddpm': TabDDPMWrapper}

def main():
    # Set random seeds for reproducibility
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

    parser = argparse.ArgumentParser(description='Tune model hyperparameters')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['ctgan', 'gmm', 'bayesian_network', 'tabddpm'])
    parser.add_argument('--trials', type=int, default=12, help='Number of Optuna trials (default: 12)')
    parser.add_argument('--config', type=str, default='config/models.yaml')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed')
    parser.add_argument('--output-dir', type=str, default='outputs/models')
    
    args = parser.parse_args()
    logger = setup_logger('tune', console=True)
    logger.info(f"Tuning {args.model} on {args.dataset} with {args.trials} Optuna trials")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['models'][args.model]
    param_space = create_param_space_from_config(model_config)
    
    data = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}_preprocessed.csv"))
    train_size = int(0.8 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    # Save validation set as test set for reproducibility
    test_file = os.path.join(args.data_dir, f"{args.dataset}_test.csv")
    val_data.to_csv(test_file, index=False)

    with open(os.path.join('data/info', f"{args.dataset}_info.json"), 'r') as f:
        feature_info = json.load(f)
    
    loader = DataLoader()
    _, structure = loader.load_dataset(args.dataset)
    
    tuner = OptunaTuner(MODEL_CLASSES[args.model], args.model, n_trials=args.trials)
    tuner.feature_info = feature_info
    tuner.graph_path = os.path.join('benchmarks_with_ground_truth/txt', f"{args.dataset}.txt")
    results = tuner.tune(train_data, val_data, structure, param_space,
                        categorical_columns=feature_info['categorical_features'] if args.model == 'ctgan' else None)
    
    os.makedirs(args.output_dir, exist_ok=True)
    study_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_study.pkl")
    tuner.save_study(study_file)
    
    logger.info(f"Best params: {results['best_params']}")
    logger.info(f"Best score: {results['best_value']}")

    # Save the best Optuna score (not the retrained value) for visualization
    with open(os.path.join(args.output_dir, f"{args.dataset}_{args.model}_best_ci_auc.json"), 'w') as f:
        json.dump({'ci_auc': results['best_value']}, f, indent=2)

    # After tuning, save the best model and compute CI AUC on synthetic data
    best_model = MODEL_CLASSES[args.model](**results['best_params'])
    if args.model == 'ctgan':
        best_model.fit(train_data, categorical_columns=feature_info['categorical_features'])
    else:
        best_model.fit(train_data)
    # Save best model
    best_model_path = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_best.pkl")
    best_model.save(best_model_path)
    # Generate synthetic data for validation set size
    synthetic_data = best_model.sample(len(val_data))
    synthetic_data = synthetic_data[val_data.columns]
    # Save synthetic data to a dedicated file for downstream evaluation
    synthetic_best_path = os.path.join('outputs/synthetic', f"{args.dataset}_{args.model}_synthetic_best.csv")
    os.makedirs(os.path.dirname(synthetic_best_path), exist_ok=True)
    synthetic_data.to_csv(synthetic_best_path, index=False)
    print(f"[INFO] Saved best synthetic data to: {synthetic_best_path}")
    # Compute CI AUC on synthetic data (for reference, not for visualization)
    ci_auc_tuple = compute_ci_auc(synthetic_data, feature_info, os.path.join('benchmarks_with_ground_truth/txt', f"{args.dataset}.txt"))
    ci_auc = ci_auc_tuple[0]
    print(f"[INFO] Best model CI ROC AUC on synthetic data: {ci_auc:.4f}")
    logger.info(f"Best model CI ROC AUC on synthetic data: {ci_auc:.4f}")
    logger.info("Tuning complete!")

if __name__ == '__main__':
    main()
