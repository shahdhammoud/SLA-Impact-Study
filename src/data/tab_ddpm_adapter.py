"""
Adapter to convert data to tab-ddpm format.

This module bridges our data format with the tab-ddpm framework requirements.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Optional


class TabDDPMAdapter:
    """
    Adapter for tab-ddpm framework data format.
    
    Tab-ddpm expects:
    - X_num: Numerical features (continuous)
    - X_cat: Categorical features (encoded as integers)
    - y: Target variable
    - info.json: Metadata about the dataset
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize adapter.
        
        Args:
            output_dir: Directory to save tab-ddpm formatted data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def convert_dataset(self, df: pd.DataFrame, 
                       categorical_features: list,
                       continuous_features: list,
                       target_column: Optional[str] = None,
                       dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Convert dataset to tab-ddpm format.
        
        Args:
            df: Input dataframe (already preprocessed)
            categorical_features: List of categorical feature names
            continuous_features: List of continuous feature names
            target_column: Name of target column (defaults to last column)
            dataset_name: Name for the dataset
            
        Returns:
            Dictionary with paths to saved files
        """
        if target_column is None:
            target_column = df.columns[-1]
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_column]
        
        # Get categorical and continuous columns (excluding target)
        cat_cols = [col for col in categorical_features if col != target_column and col in feature_cols]
        num_cols = [col for col in continuous_features if col != target_column and col in feature_cols]
        
        # Prepare data arrays
        X_cat = df[cat_cols].values if cat_cols else None
        X_num = df[num_cols].values if num_cols else None
        y = df[target_column].values
        
        # Save data
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        if X_cat is not None and len(cat_cols) > 0:
            np.save(os.path.join(dataset_dir, 'X_cat.npy'), X_cat.astype(int))
        
        if X_num is not None and len(num_cols) > 0:
            np.save(os.path.join(dataset_dir, 'X_num.npy'), X_num.astype(float))
        
        np.save(os.path.join(dataset_dir, 'y.npy'), y)
        
        # Create info.json metadata
        info = {
            'name': dataset_name,
            'task_type': self._infer_task_type(y),
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'n_categorical': len(cat_cols),
            'n_continuous': len(num_cols),
            'categorical_features': cat_cols,
            'continuous_features': num_cols,
            'target_column': target_column,
            'cat_cardinalities': [int(df[col].nunique()) for col in cat_cols] if cat_cols else [],
        }
        
        with open(os.path.join(dataset_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        return {
            'dataset_dir': dataset_dir,
            'X_cat_path': os.path.join(dataset_dir, 'X_cat.npy') if X_cat is not None else None,
            'X_num_path': os.path.join(dataset_dir, 'X_num.npy') if X_num is not None else None,
            'y_path': os.path.join(dataset_dir, 'y.npy'),
            'info_path': os.path.join(dataset_dir, 'info.json'),
            'info': info
        }
    
    def _infer_task_type(self, y: np.ndarray) -> str:
        """
        Infer task type from target variable.
        
        Args:
            y: Target variable array
            
        Returns:
            'regression' or 'binclass' or 'multiclass'
        """
        n_unique = len(np.unique(y))
        
        if n_unique == 2:
            return 'binclass'
        elif n_unique <= 10:
            return 'multiclass'
        else:
            return 'regression'
    
    def load_tabddpm_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load a dataset in tab-ddpm format.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with loaded data and metadata
        """
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Load info
        with open(os.path.join(dataset_dir, 'info.json'), 'r') as f:
            info = json.load(f)
        
        # Load data
        result = {'info': info}
        
        X_cat_path = os.path.join(dataset_dir, 'X_cat.npy')
        if os.path.exists(X_cat_path):
            result['X_cat'] = np.load(X_cat_path)
        
        X_num_path = os.path.join(dataset_dir, 'X_num.npy')
        if os.path.exists(X_num_path):
            result['X_num'] = np.load(X_num_path)
        
        result['y'] = np.load(os.path.join(dataset_dir, 'y.npy'))
        
        return result
    
    def convert_back_to_dataframe(self, dataset_name: str) -> pd.DataFrame:
        """
        Convert tab-ddpm format back to pandas DataFrame.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with the data
        """
        data = self.load_tabddpm_dataset(dataset_name)
        info = data['info']
        
        df_parts = []
        
        # Add categorical features
        if 'X_cat' in data and info['n_categorical'] > 0:
            cat_df = pd.DataFrame(
                data['X_cat'], 
                columns=info['categorical_features']
            )
            df_parts.append(cat_df)
        
        # Add continuous features
        if 'X_num' in data and info['n_continuous'] > 0:
            num_df = pd.DataFrame(
                data['X_num'], 
                columns=info['continuous_features']
            )
            df_parts.append(num_df)
        
        # Add target
        target_df = pd.DataFrame(
            data['y'], 
            columns=[info['target_column']]
        )
        df_parts.append(target_df)
        
        # Combine all parts
        df = pd.concat(df_parts, axis=1)
        
        return df
