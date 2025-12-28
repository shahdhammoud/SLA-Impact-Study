import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling

from .base import BaseGenerativeModel


class BayesianNetworkWrapper(BaseGenerativeModel):

    def __init__(self, **kwargs):

        super().__init__(model_name="BayesianNetwork", **kwargs)
        
        default_params = {
            'learning_algorithm': 'hillclimb',
            'scoring_method': 'bic',
            'max_iter': 1000,
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        self.bn_model = None
        self.sampler = None
    
    def fit(self, data: pd.DataFrame, **kwargs):

        if self.params['scoring_method'] == 'bic':
            scoring = BicScore(data)
        elif self.params['scoring_method'] == 'k2':
            scoring = K2Score(data)
        else:
            scoring = BicScore(data)
        
        if self.params['learning_algorithm'] == 'hillclimb':
            estimator = HillClimbSearch(data)
            learned_structure = estimator.estimate(
                scoring_method=scoring,
                max_iter=self.params['max_iter']
            )
        else:
            estimator = HillClimbSearch(data)
            learned_structure = estimator.estimate(
                scoring_method=scoring,
                max_iter=self.params['max_iter']
            )
        
        self.bn_model = BayesianNetwork(learned_structure.edges())
        
        self.bn_model.fit(data, estimator=MaximumLikelihoodEstimator)
        
        self.sampler = BayesianModelSampling(self.bn_model)
        
        self.model = self.bn_model
        self.is_fitted = True
        
        final_score = scoring.score(learned_structure)
        self.training_losses = [final_score]
        
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'n_edges': len(self.bn_model.edges()),
            'edges': list(self.bn_model.edges()),
            'final_score': final_score,
            'training_losses': self.training_losses
        }
    
    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples from the Bayesian Network.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            DataFrame with synthetic samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        synthetic_data = self.sampler.forward_sample(size=n_samples)
        
        synthetic_data = synthetic_data[self.metadata['features']]
        
        return synthetic_data
    
    def get_structure(self):
        """
        Get the learned Bayesian Network structure.
        
        Returns:
            List of edges
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return list(self.bn_model.edges())
    
    def get_cpds(self):
        """
        Get Conditional Probability Distributions.
        
        Returns:
            List of CPDs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.bn_model.get_cpds()
