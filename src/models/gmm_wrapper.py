"""
Gaussian Mixture Model wrapper for tabular data generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.mixture import GaussianMixture

from .base import BaseGenerativeModel


class GMMWrapper(BaseGenerativeModel):
    """Wrapper for Gaussian Mixture Model."""
    
    def __init__(self, **kwargs):
        """
        Initialize GMM model.
        
        Args:
            **kwargs: GMM parameters
        """
        super().__init__(model_name="GMM", **kwargs)
        
        # Default parameters
        default_params = {
            'n_components': 10,
            'covariance_type': 'full',
            'max_iter': 100,
            'n_init': 10,
            'random_state': 42
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit GMM to data.
        
        Args:
            data: Training data (should be numerical)
            **kwargs: Additional training parameters
        """
        # Convert to numpy array
        X = data.values
        
        # Initialize GMM
        self.model = GaussianMixture(
            n_components=self.params['n_components'],
            covariance_type=self.params['covariance_type'],
            max_iter=self.params['max_iter'],
            n_init=self.params['n_init'],
            random_state=self.params['random_state']
        )
        
        # Fit model
        self.model.fit(X)
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_
        }
    
    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            DataFrame with synthetic samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        # Generate samples
        X_synthetic, _ = self.model.sample(n_samples)
        
        # Convert to DataFrame
        synthetic_data = pd.DataFrame(
            X_synthetic,
            columns=self.metadata['features']
        )
        
        return synthetic_data
    
    def score(self, data: pd.DataFrame) -> float:
        """
        Compute log-likelihood of data under the model.
        
        Args:
            data: Data to score
            
        Returns:
            Average log-likelihood per sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X = data.values
        return self.model.score(X)
