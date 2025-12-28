#!/usr/bin/env python3
"""
Script 1: Preprocess dataset.

Loads raw CSV and TXT files, detects feature types, preprocesses data,
and converts to tab-ddpm format.
"""

import argparse
import os
import sys
import yaml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.tab_ddpm_adapter import TabDDPMAdapter
from src.utils.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for generative modeling')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Name of the dataset (without extension)')
    parser.add_argument('--config', type=str, default='config/datasets.yaml',
                       help='Path to dataset configuration file')
    parser.add_argument('--output-dir', type=str, default='data/preprocessed',
                       help='Output directory for preprocessed data')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize continuous features')
    
    args = parser.parse_args()
    
    logger = setup_logger('preprocess', console=True)
    
    logger.info(f"Preprocessing dataset: {args.dataset}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    processing_config = config.get('processing', {})
    
    # Load data
    logger.info("Loading raw data...")
    loader = DataLoader()
    
    try:
        data, structure = loader.load_dataset(args.dataset)
        logger.info(f"Loaded data: {len(data)} samples, {len(data.columns)} features")
        logger.info(f"Loaded structure: {structure.number_of_nodes()} nodes, {structure.number_of_edges()} edges")
        if data.empty:
            logger.error(f"Loaded data is empty for dataset: {args.dataset}. Check the raw CSV file and loader logic.")
            sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)
    
    # Detect and preprocess
    logger.info("Detecting feature types...")
    preprocessor = DataPreprocessor(
        categorical_threshold=processing_config.get('categorical_threshold', 10)
    )
    
    preprocessor.fit(data)
    feature_info = preprocessor.get_feature_info()
    
    logger.info(f"Detected {feature_info['n_categorical']} categorical features: {feature_info['categorical_features']}")
    logger.info(f"Detected {feature_info['n_continuous']} continuous features: {feature_info['continuous_features']}")
    
    # Transform data
    logger.info("Transforming data...")
    data_transformed = preprocessor.transform(data, normalize=args.normalize)
    logger.info(f"Transformed data shape: {data_transformed.shape}")
    if data_transformed.empty or data_transformed.shape[0] == 0 or data_transformed.shape[1] == 0:
        logger.error(f"Transformed data is empty for dataset: {args.dataset}. Check preprocessing logic.")
        sys.exit(1)

    # Save metadata
    metadata_dir = 'data/info'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_file = os.path.join(metadata_dir, f"{args.dataset}_info.json")
    preprocessor.save_metadata(metadata_file)
    logger.info(f"Saved metadata to {metadata_file}")
    
    # Convert to tab-ddpm format
    logger.info("Converting to tab-ddpm format...")
    adapter = TabDDPMAdapter(args.output_dir)
    
    result = adapter.convert_dataset(
        data_transformed,
        categorical_features=preprocessor.categorical_features,
        continuous_features=preprocessor.continuous_features,
        dataset_name=args.dataset
    )
    
    logger.info(f"Saved tab-ddpm format to {result['dataset_dir']}")
    logger.info(f"  - X_cat: {result['X_cat_path']}")
    logger.info(f"  - X_num: {result['X_num_path']}")
    logger.info(f"  - y: {result['y_path']}")
    logger.info(f"  - info: {result['info_path']}")
    
    # Save preprocessed CSV for direct use
    csv_output = os.path.join(args.output_dir, f"{args.dataset}_preprocessed.csv")
    data_transformed.to_csv(csv_output, index=False)
    logger.info(f"Saved preprocessed CSV to {csv_output}")
    
    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()
