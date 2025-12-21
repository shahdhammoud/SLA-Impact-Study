import pandas as pd
import numpy as np
import json
import os
import pickle # Added for loading transformers
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder

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
        
        return df
