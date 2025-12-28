#!/usr/bin/env python3
"""
Script 2: Train generative model.

Trains a specified generative model on preprocessed data.
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import json
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ctgan_wrapper import CTGANWrapper
from src.models.gmm_wrapper import GMMWrapper
from src.models.bayesian_network import BayesianNetworkWrapper
from src.models.tabddpm_wrapper import TabDDPMWrapper
from src.utils.logging_utils import setup_logger
from src.utils.visualization_utils import plot_training_loss


# Set random seed for reproducibility
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


MODEL_CLASSES = {
    'ctgan': CTGANWrapper,
    'gmm': GMMWrapper,
    'bayesian_network': BayesianNetworkWrapper,
    'tabddpm': TabDDPMWrapper
}


def main():
    parser = argparse.ArgumentParser(description='Train generative model')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--model', type=str, required=True,
                       choices=['ctgan', 'gmm', 'bayesian_network', 'tabddpm'],
                       help='Model to train')
    parser.add_argument('--config', type=str, default='config/models.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed',
                       help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='outputs/models',
                       help='Output directory for trained models')
    parser.add_argument('--use-best-params', action='store_true',
                       help='Use best parameters from tuning')
    parser.add_argument('--params-file', type=str, default=None,
                       help='Path to custom parameters JSON file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (for CTGAN/TabDDPM)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')

    args = parser.parse_args()
    
    logger = setup_logger('train', console=True)
    
    logger.info(f"Training model: {args.model} on dataset: {args.dataset}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['models'][args.model]
    
    # Load parameters
    if args.use_best_params:
        params_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_best_params.json")
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            logger.info(f"Using best parameters from {params_file}")
        else:
            logger.warning(f"Best parameters file not found: {params_file}. Using default parameters.")
            params = model_config['default_params']
    elif args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
        logger.info(f"Using custom parameters from {args.params_file}")
    else:
        params = model_config['default_params']
        logger.info("Using default parameters")
    
    # Apply command-line overrides
    if args.epochs is not None:
        params['epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    data_file = os.path.join(args.data_dir, f"{args.dataset}_preprocessed.csv")
    data = pd.read_csv(data_file)
    logger.info(f"Loaded {len(data)} samples")

    # Load feature info
    info_file = os.path.join('data/info', f"{args.dataset}_info.json")
    with open(info_file, 'r') as f:
        feature_info = json.load(f)

    categorical_features = feature_info['categorical_features']

    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    model_class = MODEL_CLASSES[args.model]
    model = model_class(**params)

    # Train model
    logger.info("Training model...")
    if args.model == 'ctgan':
        model.fit(data, categorical_columns=categorical_features)
    else:
        model.fit(data)

    logger.info("Training complete!")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}.pkl")
    model.save(model_file)
    logger.info(f"Saved model to {model_file}")
    
    # Save and plot training losses
    training_losses = model.get_training_losses()
    if args.model == 'ctgan' and isinstance(training_losses, dict):
        from src.utils.visualization_utils import plot_gan_losses
        gan_loss_plot_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_gen_disc_loss.png")
        plot_gan_losses(
            losses_dict=training_losses,
            model_name=args.model.upper(),
            save_path=gan_loss_plot_file,
            title=f"{args.model.upper()} Generator & Discriminator Loss - {args.dataset}"
        )
        # Save both losses to JSON
        loss_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_gen_disc_losses.json")
        with open(loss_file, 'w') as f:
            json.dump(training_losses, f, indent=2)
        logger.info(f"Saved generator/discriminator losses to {loss_file}")
    elif training_losses:
        # Save loss plot
        loss_plot_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_training_loss.png")
        plot_training_loss(
            losses=training_losses,
            model_name=args.model.upper(),
            save_path=loss_plot_file,
            title=f"{args.model.upper()} Training Loss - {args.dataset}"
        )
        # Save losses to JSON
        loss_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model}_training_losses.json")
        with open(loss_file, 'w') as f:
            json.dump({'losses': training_losses}, f, indent=2)
        logger.info(f"Saved training losses to {loss_file}")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
