"""
Data preprocessor with automatic feature type detection.

Handles categorical and continuous features appropriately for downstream models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
import json
import os


class DataPreprocessor:
    """Preprocess datasets with automatic categorical/continuous detection."""
    
    def __init__(self, categorical_threshold: int = 10):
        """
        Initialize preprocessor.
        
        Args:
            categorical_threshold: Columns with <= this many unique values
                                  are treated as categorical
        """
        self.categorical_threshold = categorical_threshold
        self.categorical_features = []
        self.continuous_features = []
        self.label_encoders = {}
        self.scaler = None
        self.feature_info = {}
    
    def detect_feature_types(self, df: pd.DataFrame, 
                            target_column: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Automatically detect categorical and continuous features.
        
        Args:
            df: Input dataframe
            target_column: Name of target column (defaults to last column)
            
        Returns:
            Dictionary with 'categorical' and 'continuous' feature lists
        """
        if target_column is None:
            target_column = df.columns[-1]
        
        # Exclude target from feature detection
        features = [col for col in df.columns if col != target_column]
        
        categorical = []
        continuous = []
        
        for col in features:
            # Check if already object/string type
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical.append(col)
            # Check number of unique values
            elif df[col].nunique() <= self.categorical_threshold:
                categorical.append(col)
            else:
                continuous.append(col)
        
        # Always include target in the appropriate category
        if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
            categorical.append(target_column)
        elif df[target_column].nunique() <= self.categorical_threshold:
            categorical.append(target_column)
        else:
            continuous.append(target_column)
        
        self.categorical_features = categorical
        self.continuous_features = continuous
        
        return {
            'categorical': categorical,
            'continuous': continuous
        }
    
    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'DataPreprocessor':
        """
        Fit preprocessor to data.
        
        Args:
            df: Input dataframe
            target_column: Name of target column (defaults to last column)
            
        Returns:
            Self for chaining
        """
        # Detect feature types
        feature_types = self.detect_feature_types(df, target_column)
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
            
            # Store feature info
            self.feature_info[col] = {
                'type': 'categorical',
                'n_categories': len(le.classes_),
                'categories': le.classes_.tolist()
            }
        
        # Fit scaler for continuous features
        if self.continuous_features:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.continuous_features])
            
            for col in self.continuous_features:
                self.feature_info[col] = {
                    'type': 'continuous',
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return self
    
    def transform(self, df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Input dataframe
            normalize: Whether to normalize continuous features
            
        Returns:
            Transformed dataframe
        """
        df_transformed = df.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in df_transformed.columns:
                df_transformed[col] = self.label_encoders[col].transform(
                    df_transformed[col].astype(str)
                )
        
        # Normalize continuous features
        if normalize and self.continuous_features and self.scaler is not None:
            continuous_cols = [col for col in self.continuous_features if col in df_transformed.columns]
            if continuous_cols:
                df_transformed[continuous_cols] = self.scaler.transform(
                    df_transformed[continuous_cols]
                )
        
        return df_transformed
    
    def inverse_transform(self, df: pd.DataFrame, denormalize: bool = True) -> pd.DataFrame:
        """
        Inverse transform data back to original format.
        
        Args:
            df: Transformed dataframe
            denormalize: Whether to denormalize continuous features
            
        Returns:
            Data in original format
        """
        df_original = df.copy()
        
        # Decode categorical features
        for col in self.categorical_features:
            if col in df_original.columns:
                df_original[col] = self.label_encoders[col].inverse_transform(
                    df_original[col].astype(int)
                )
        
        # Denormalize continuous features
        if denormalize and self.continuous_features and self.scaler is not None:
            continuous_cols = [col for col in self.continuous_features if col in df_original.columns]
            if continuous_cols:
                df_original[continuous_cols] = self.scaler.inverse_transform(
                    df_original[continuous_cols]
                )
        
        return df_original
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None,
                     normalize: bool = True) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            normalize: Whether to normalize continuous features
            
        Returns:
            Transformed dataframe
        """
        self.fit(df, target_column)
        return self.transform(df, normalize)
    
    def save_metadata(self, filepath: str):
        """
        Save preprocessor metadata to file.
        
        Args:
            filepath: Path to save metadata JSON
        """
        metadata = {
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'feature_info': self.feature_info,
            'categorical_threshold': self.categorical_threshold
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, filepath: str):
        """
        Load preprocessor metadata from file.
        
        Args:
            filepath: Path to metadata JSON
        """
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        self.categorical_features = metadata['categorical_features']
        self.continuous_features = metadata['continuous_features']
        self.feature_info = metadata['feature_info']
        self.categorical_threshold = metadata['categorical_threshold']
    
    def get_feature_info(self) -> Dict:
        """
        Get information about all features.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'n_features': len(self.categorical_features) + len(self.continuous_features),
            'n_categorical': len(self.categorical_features),
            'n_continuous': len(self.continuous_features),
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'feature_info': self.feature_info
        }
