from src.models.base import BaseGenerativeModel
from sklearn.mixture import GaussianMixture
import pickle

class GMMWrapper(BaseGenerativeModel):
    def __init__(self, **kwargs):
        # Remove 'model_name' from kwargs if it exists, as we are explicitly setting it
        kwargs.pop('model_name', None)
        super().__init__(model_name="GMM", **kwargs)
        self.model = None

    def fit(self, data):
        self.model = GaussianMixture(**self.params)
        self.model.fit(data)

    def sample(self, n_samples):
        return self.model.sample(n_samples)[0]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
