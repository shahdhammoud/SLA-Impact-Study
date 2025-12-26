"""
GES (Greedy Equivalence Search) algorithm implementation using causal-learn.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

from causallearn.search.ScoreBased.GES import ges

from .base import BaseStructureLearner


class GESLearner(BaseStructureLearner):
    """GES algorithm for causal structure learning."""
    
    def __init__(self, **kwargs):
        """
        Initialize GES learner.
        
        Args:
            **kwargs: GES algorithm parameters
        """
        super().__init__(algorithm_name="GES", **kwargs)
        
        # Default parameters
        default_params = {
            'score_func': 'local_score_BIC',
            'maxP': None,
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure using GES algorithm.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Learned causal graph
        """
        # Convert to numpy array
        X = data.values.astype(np.float64)
        node_names = list(data.columns)
        
        # Check for zero-variance columns and add small noise to prevent singular matrix
        variances = np.var(X, axis=0)
        zero_var_cols = np.where(variances < 1e-10)[0]
        if len(zero_var_cols) > 0:
            # Add tiny noise to zero-variance columns to prevent singular matrix
            for col_idx in zero_var_cols:
                X[:, col_idx] = X[:, col_idx] + np.random.normal(0, 1e-6, size=X.shape[0])

        # Run GES algorithm
        try:
            record = ges(
                X,
                score_func=self.params['score_func'],
                maxP=self.params['maxP']
            )
        except np.linalg.LinAlgError as e:
            # If still singular, return empty graph
            print(f"GES encountered singular matrix error: {e}. Returning empty graph.")
            self.learned_graph = nx.DiGraph()
            self.learned_graph.add_nodes_from(node_names)
            self.is_fitted = True
            self.metadata = {
                'n_samples': len(data),
                'n_features': len(node_names),
                'n_edges': 0,
                'features': node_names,
                'error': str(e)
            }
            return self.learned_graph

        # Convert to NetworkX graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        # Extract edges from causal graph
        adj_matrix = record['G'].graph
        n_nodes = len(node_names)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
                    # Directed edge i -> j
                    self.learned_graph.add_edge(node_names[i], node_names[j])
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    # Directed edge j -> i
                    self.learned_graph.add_edge(node_names[j], node_names[i])
        
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names
        }
        
        return self.learned_graph
