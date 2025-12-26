"""
CTGAN model wrapper (original, compatible version).
"""

import pandas as pd
from typing import Optional
import torch
from ctgan import CTGAN

from .base import BaseGenerativeModel


class CTGANWrapper(BaseGenerativeModel):
    """Wrapper for CTGAN generative model (original implementation)."""

    def __init__(self, **kwargs):
        super().__init__(model_name="CTGAN", **kwargs)
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
            'verbose': True
        }
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

    def fit(self, data: pd.DataFrame, categorical_columns: Optional[list] = None, **kwargs):
        """
        Fit CTGAN to data (original implementation).
        """
        if categorical_columns is None:
            categorical_columns = []
        self.model = CTGAN(
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            generator_dim=self.params['generator_dim'],
            discriminator_dim=self.params['discriminator_dim'],
            generator_lr=self.params['generator_lr'],
            discriminator_lr=self.params['discriminator_lr'],
            discriminator_steps=self.params['discriminator_steps'],
            pac=self.params['pac'],
            verbose=self.params['verbose']
        )
        self.model.fit(data, discrete_columns=categorical_columns)
        self.is_fitted = True
        self.training_losses = []  # Not available in this implementation
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'categorical_columns': categorical_columns or [],
            'training_losses': self.training_losses
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

    def get_training_losses(self):
        # Not available in this implementation
        return []
