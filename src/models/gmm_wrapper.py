from src.models.base import BaseGenerativeModel
from sklearn.mixture import GaussianMixture
import pickle
import pandas as pd
import numpy as np

class GMMWrapper(BaseGenerativeModel):
    def __init__(self, **kwargs):
        # Remove 'model_name' from kwargs if it exists, as we are explicitly setting it
        kwargs.pop('model_name', None)
        super().__init__(model_name="GMM", **kwargs)
        self.model = None
        self.column_names = None  # Store column names for output

    def fit(self, data):
        self.model = GaussianMixture(**self.params)
        self.model.fit(data)
        self.is_fitted = True
        self.column_names = list(data.columns)  # Store column names
        # Track convergence (GMM doesn't have epochs, but we can track iterations)
        # Store the log likelihood as a proxy for "loss"
        if hasattr(self.model, 'lower_bound_'):
            # GMM converged, store the final log likelihood
            self.training_losses = [self.model.lower_bound_]
        else:
            self.training_losses = []
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': data.shape[1],
            'n_components': self.params.get('n_components', 10),
            'converged': self.model.converged_,
            'n_iter': self.model.n_iter_,
            'training_losses': self.training_losses
        }

    def sample(self, n_samples):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        samples, _ = self.model.sample(n_samples)
        # Use stored column names if available
        if self.column_names is not None and len(self.column_names) == samples.shape[1]:
            df = pd.DataFrame(samples, columns=self.column_names)
        else:
            df = pd.DataFrame(samples)
        # Ensure the output has exactly n_samples rows
        if len(df) != n_samples:
            # Try to resample or pad/truncate as needed
            if len(df) > n_samples:
                df = df.iloc[:n_samples]
            else:
                # If too few, repeat sampling
                extra, _ = self.model.sample(n_samples - len(df))
                if self.column_names is not None and len(self.column_names) == extra.shape[1]:
                    extra_df = pd.DataFrame(extra, columns=self.column_names)
                else:
                    extra_df = pd.DataFrame(extra)
                df = pd.concat([df, extra_df], ignore_index=True)
        return df

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True

    def get_training_losses(self):
        """Return training losses (log-likelihood) for plotting."""
        return getattr(self, 'training_losses', [])
