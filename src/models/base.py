"""
Base interface for generative models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
import joblib


class BaseGenerativeModel(ABC):
    """Abstract base class for generative models."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize model.
        
        Args:
            model_name: Name of the model
            **kwargs: Model-specific parameters
        """
        self.model_name = model_name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        self.metadata = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit the model to data.
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            DataFrame with synthetic samples
        """
        pass
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_name': self.model_name,
            'params': self.params,
            'model': self.model,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        save_dict = joblib.load(filepath)
        
        self.model_name = save_dict['model_name']
        self.params = save_dict['params']
        self.model = save_dict['model']
        self.metadata = save_dict['metadata']
        self.is_fitted = save_dict['is_fitted']
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        self.params.update(params)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary of metadata
        """
        return self.metadata.copy()
