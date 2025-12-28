from src.models.base import BaseGenerativeModel
from sklearn.mixture import GaussianMixture
import pickle
import pandas as pd
import numpy as np

class GMMWrapper(BaseGenerativeModel):
    def __init__(self, **kwargs):
        kwargs.pop('model_name', None)
        super().__init__(model_name="GMM", **kwargs)
        self.model = None
        self.column_names = None

    def fit(self, data):
        self.model = GaussianMixture(**self.params)
        self.model.fit(data)
        self.is_fitted = True
        self.column_names = list(data.columns)
        if hasattr(self.model, 'lower_bound_'):
            self.training_losses = [self.model.lower_bound_]
        else:
            self.training_losses = []
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
        if self.column_names is not None and len(self.column_names) == samples.shape[1]:
            df = pd.DataFrame(samples, columns=self.column_names)
        else:
            df = pd.DataFrame(samples)
        if len(df) != n_samples:
            if len(df) > n_samples:
                df = df.iloc[:n_samples]
            else:
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
        return getattr(self, 'training_losses', [])
