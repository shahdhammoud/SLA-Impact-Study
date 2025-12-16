"""
CTGAN model wrapper using SDV library.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import torch

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from .base import BaseGenerativeModel


class CTGANWrapper(BaseGenerativeModel):
    """Wrapper for CTGAN generative model."""
    
    def __init__(self, **kwargs):
        """
        Initialize CTGAN model.
        
        Args:
            **kwargs: CTGAN parameters
        """
        super().__init__(model_name="CTGAN", **kwargs)
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() and kwargs.get('use_gpu', True) else 'cpu'
        
        # Default parameters
        default_params = {
            'epochs': 300,
            'batch_size': 500,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256),
            'generator_lr': 2e-4,
            'discriminator_lr': 2e-4,
            'discriminator_steps': 1,
            'pac': 10,
            'cuda': self.device == 'cuda'
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, categorical_columns: Optional[list] = None, **kwargs):
        """
        Fit CTGAN to data.
        
        Args:
            data: Training data
            categorical_columns: List of categorical column names
            **kwargs: Additional training parameters
        """
        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        
        # Update with categorical columns if provided
        if categorical_columns:
            for col in categorical_columns:
                if col in data.columns:
                    metadata.update_column(col, sdtype='categorical')
        
        # Initialize synthesizer
        self.model = CTGANSynthesizer(
            metadata=metadata,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            generator_dim=self.params['generator_dim'],
            discriminator_dim=self.params['discriminator_dim'],
            generator_lr=self.params['generator_lr'],
            discriminator_lr=self.params['discriminator_lr'],
            discriminator_steps=self.params['discriminator_steps'],
            pac=self.params['pac'],
            cuda=self.params['cuda']
        )
        
        # Fit model
        self.model.fit(data)
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'categorical_columns': categorical_columns or []
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
        
        synthetic_data = self.model.sample(n_samples)
        
        return synthetic_data
