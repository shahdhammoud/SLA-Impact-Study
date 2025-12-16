"""
Bayesian Network generative model using pgmpy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling

from .base import BaseGenerativeModel


class BayesianNetworkWrapper(BaseGenerativeModel):
    """Wrapper for Bayesian Network generative model."""
    
    def __init__(self, **kwargs):
        """
        Initialize Bayesian Network model.
        
        Args:
            **kwargs: BN parameters
        """
        super().__init__(model_name="BayesianNetwork", **kwargs)
        
        # Default parameters
        default_params = {
            'learning_algorithm': 'hillclimb',
            'scoring_method': 'bic',
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        self.bn_model = None
        self.sampler = None
    
    def fit(self, data: pd.DataFrame, **kwargs):
        """
        Fit Bayesian Network to data.
        
        This learns both structure and parameters.
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
        """
        # Structure learning
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
            # Fallback to hill climb
            estimator = HillClimbSearch(data)
            learned_structure = estimator.estimate(
                scoring_method=scoring,
                max_iter=self.params['max_iter']
            )
        
        # Create Bayesian Network with learned structure
        self.bn_model = BayesianNetwork(learned_structure.edges())
        
        # Parameter learning (CPDs)
        self.bn_model.fit(data, estimator=MaximumLikelihoodEstimator)
        
        # Create sampler
        self.sampler = BayesianModelSampling(self.bn_model)
        
        self.model = self.bn_model
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'n_edges': len(self.bn_model.edges()),
            'edges': list(self.bn_model.edges())
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
        
        # Generate samples using forward sampling
        synthetic_data = self.sampler.forward_sample(size=n_samples)
        
        # Ensure column order matches original
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
