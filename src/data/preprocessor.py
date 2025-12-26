import pandas as pd
import numpy as np
import json
import os
import pickle # Added for loading transformers
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder


class DataPreprocessor:
    """Data preprocessor for detecting and transforming features."""

    def __init__(self, categorical_threshold=10):
        """
        Initialize preprocessor.

        Args:
            categorical_threshold: Max unique values to consider a feature categorical
        """
        self.categorical_threshold = categorical_threshold
        self.categorical_features = []
        self.continuous_features = []
        self.scaler = None
        self.encoders = {}
        self.feature_info = {}

    def fit(self, data: pd.DataFrame):
        """
        Fit the preprocessor to the data.

        Args:
            data: Input dataframe
        """
        # Detect feature types
        for col in data.columns:
            n_unique = data[col].nunique()
            if n_unique <= self.categorical_threshold or data[col].dtype == 'object':
                self.categorical_features.append(col)
            else:
                self.continuous_features.append(col)

        # Fit transformers for continuous features
        if self.continuous_features:
            self.scaler = StandardScaler()
            self.scaler.fit(data[self.continuous_features])

        # Fit encoders for categorical features
        for col in self.categorical_features:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(data[[col]])
            self.encoders[col] = encoder

        # Store feature info
        self.feature_info = {
            'n_categorical': len(self.categorical_features),
            'n_continuous': len(self.continuous_features),
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'total_features': len(data.columns),
            'n_samples': len(data)
        }

    def transform(self, data: pd.DataFrame, normalize=True):
        """
        Transform the data.

        Args:
            data: Input dataframe
            normalize: Whether to normalize continuous features

        Returns:
            Transformed dataframe
        """
        data_transformed = data.copy()

        # Transform continuous features
        if normalize and self.continuous_features and self.scaler:
            data_transformed[self.continuous_features] = self.scaler.transform(data[self.continuous_features])

        # Transform categorical features
        for col in self.categorical_features:
            if col in self.encoders:
                data_transformed[col] = self.encoders[col].transform(data[[col]]).flatten()

        return data_transformed

    def get_feature_info(self):
        """Get feature information."""
        return self.feature_info

    def save_metadata(self, filepath):
        """
        Save metadata to file.

        Args:
            filepath: Path to save metadata
        """
        metadata = {
            'categorical_features': self.categorical_features,
            'continuous_features': self.continuous_features,
            'feature_info': self.feature_info
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)


class Preprocessor:
    def __init__(self, dataset, model=None):
        self.dataset = dataset
        self.model = model
        self.data_path = os.path.join('data', 'preprocessed', self.dataset)
        self.info = self._load_info()
        
        self.column_names = self.info.get('column_names', [])
        self.numerical_features = self.info.get('continuous_features', [])
        self.categorical_features = self.info.get('categorical_features', [])

        self.numerical_transformer = None
        self.categorical_transformer = {}

        numerical_transformer_path = os.path.join(self.data_path, 'numerical_transformer.pkl')
        categorical_transformer_path = os.path.join(self.data_path, 'categorical_transformer.pkl')

        if os.path.exists(numerical_transformer_path):
            with open(numerical_transformer_path, 'rb') as f:
                self.numerical_transformer = pickle.load(f)
        if os.path.exists(categorical_transformer_path):
            with open(categorical_transformer_path, 'rb') as f:
                self.categorical_transformer = pickle.load(f)

    def _load_info(self):
        info_path_preprocessed = os.path.join(self.data_path, 'info.json')
        if os.path.exists(info_path_preprocessed):
            with open(info_path_preprocessed, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Info file not found for dataset {self.dataset} at {info_path_preprocessed}")

    def transform(self, df):
        transformed_df = df.copy()

        if self.numerical_features and self.numerical_transformer:
            transformed_df[self.numerical_features] = self.numerical_transformer.transform(transformed_df[self.numerical_features])

        if self.categorical_features:
            for col in self.categorical_features:
                if col in transformed_df.columns and col in self.categorical_transformer:
                    transformed_df[col] = self.categorical_transformer[col].transform(transformed_df[[col]]).flatten()

        return transformed_df.to_numpy()


    def inverse_transform(self, data, denormalize=True):
        # FIX START: Convert input numpy array to DataFrame
        if isinstance(data, np.ndarray):
            if hasattr(self, 'column_names') and self.column_names and len(self.column_names) == data.shape[1]:
                df = pd.DataFrame(data, columns=self.column_names)
            else:
                # Fallback: create DataFrame without specific column names if metadata is missing or mismatch
                df = pd.DataFrame(data)
        else:
            df = data.copy() # If it's already a DataFrame, just copy it
        # FIX END

        # Apply inverse transforms for numerical data (denormalize)
        if denormalize and self.numerical_transformer:
            # Only attempt if numerical features exist and are present in the DataFrame
            if self.numerical_features and all(col in df.columns for col in self.numerical_features):
                df[self.numerical_features] = self.numerical_transformer.inverse_transform(df[self.numerical_features])
            elif self.numerical_features:
                print(f"Warning: Some numerical features {self.numerical_features} not found in DataFrame for inverse transform. Skipping numerical inverse transformation.")

        # Apply inverse transforms for categorical data
        if self.categorical_transformer:
            for col in self.categorical_features:
                if col in df.columns and col in self.categorical_transformer:
                    df[col] = self.categorical_transformer[col].inverse_transform(df[[col]].to_numpy()).flatten()
                elif col in self.categorical_features:
                    print(f"Warning: Categorical feature {col} or its transformer not found in DataFrame for inverse transform. Skipping categorical inverse transformation for this column.")
        
        # Post-process categorical features: round and clip to valid discrete values
        # Get cardinalities if available
        cat_cardinalities = self.info.get('cat_cardinalities', [])

        for i, col in enumerate(self.categorical_features):
            if col in df.columns:
                # Get max value for this category (cardinality - 1)
                max_val = cat_cardinalities[i] - 1 if i < len(cat_cardinalities) else 1
                # Round to nearest integer and clip to valid range [0, max_category]
                df[col] = df[col].round().clip(lower=0, upper=max_val).astype(int)

        # Also handle target column if it's categorical (binary)
        target_col = self.info.get('target_column')
        if target_col and target_col in df.columns and target_col not in self.categorical_features:
            # Assume binary target
            df[target_col] = df[target_col].round().clip(lower=0, upper=1).astype(int)

        return df
