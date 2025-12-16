"""
LiNGAM (Linear Non-Gaussian Acyclic Model) algorithm implementation.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

import lingam

from .base import BaseStructureLearner


class LiNGAMLearner(BaseStructureLearner):
    """LiNGAM algorithm for causal structure learning."""
    
    def __init__(self, **kwargs):
        """
        Initialize LiNGAM learner.
        
        Args:
            **kwargs: LiNGAM algorithm parameters
        """
        super().__init__(algorithm_name="LiNGAM", **kwargs)
        
        # Default parameters
        default_params = {
            'measure': 'pwling',  # 'pwling' or 'kernel'
            'random_state': 42
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure using LiNGAM algorithm.
        
        Args:
            data: Input data (should have non-Gaussian noise)
            **kwargs: Additional parameters
            
        Returns:
            Learned causal graph
        """
        # Convert to numpy array
        X = data.values
        node_names = list(data.columns)
        
        # Run LiNGAM algorithm
        if self.params['measure'] == 'pwling':
            model = lingam.DirectLiNGAM(random_state=self.params['random_state'])
        else:
            model = lingam.DirectLiNGAM(random_state=self.params['random_state'])
        
        model.fit(X)
        
        # Get adjacency matrix
        adj_matrix = model.adjacency_matrix_
        
        # Convert to NetworkX graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        n_nodes = len(node_names)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:
                    self.learned_graph.add_edge(node_names[i], node_names[j])
        
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names,
            'causal_order': [node_names[i] for i in model.causal_order_],
            'note': 'LiNGAM assumes linear relationships and non-Gaussian noise'
        }
        
        return self.learned_graph
