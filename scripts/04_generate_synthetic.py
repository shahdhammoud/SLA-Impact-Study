import argparse
import json
import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.models import MODEL_CLASSES
from src.data.preprocessor import Preprocessor # Corrected class name

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data using a trained model.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model', type=str, required=True, help='Name of the generative model (e.g., gmm, ctgan)')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of synthetic samples to generate')

    args = parser.parse_args()

    logger.info(f"Generating {args.n_samples} samples from {args.model} trained on {args.dataset}")

    # Define paths
    output_dir = os.path.join('outputs', 'synthetic')
    model_dir = os.path.join('outputs', 'models')
    model_file = os.path.join(model_dir, f'{args.dataset}_{args.model}.pkl')
    data_dir = os.path.join('data', 'preprocessed')

    os.makedirs(output_dir, exist_ok=True)

    # Load original data to get column names
    original_data = pd.read_csv(os.path.join(data_dir, f'{args.dataset}_preprocessed.csv'))
    column_names = original_data.columns.tolist()
    logger.info(f"Loaded column names from original data: {column_names}")

    # Load the trained model
    model = MODEL_CLASSES[args.model]() # Initialize model from MODEL_CLASSES
    model.load(model_file)
    logger.info(f"Loaded model from {model_file}")

    # Generate synthetic data
    synthetic_data = model.sample(args.n_samples)

    # Ensure synthetic data has correct column names
    if synthetic_data.shape[1] == len(column_names):
        synthetic_data.columns = column_names
    logger.info(f"Generated {args.n_samples} samples")

    # Inverse transform to original scale
    preprocessor = Preprocessor(args.dataset)
    synthetic_data = preprocessor.inverse_transform(synthetic_data, denormalize=True)
    
    # Save synthetic data (without timestamp for consistent naming)
    output_filename = os.path.join(output_dir, f'{args.dataset}_{args.model}_synthetic.csv')
    synthetic_data.to_csv(output_filename, index=False)
    logger.info(f"Saved synthetic data to {output_filename}")

    logger.info("Synthetic data generation complete!")

if __name__ == '__main__':
    main()
