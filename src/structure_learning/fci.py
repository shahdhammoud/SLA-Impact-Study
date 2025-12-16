"""
FCI (Fast Causal Inference) algorithm implementation using causal-learn.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, gsq, kci

from .base import BaseStructureLearner


class FCILearner(BaseStructureLearner):
    """FCI algorithm for causal structure learning with latent confounders."""
    
    def __init__(self, **kwargs):
        """
        Initialize FCI learner.
        
        Args:
            **kwargs: FCI algorithm parameters
        """
        super().__init__(algorithm_name="FCI", **kwargs)
        
        # Default parameters
        default_params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'stable': True,
            'max_path_length': -1
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure using FCI algorithm.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Learned causal graph (PAG - Partial Ancestral Graph)
        """
        # Convert to numpy array
        X = data.values
        node_names = list(data.columns)
        
        # Select independence test
        if self.params['indep_test'] == 'fisherz':
            indep_test = fisherz
        elif self.params['indep_test'] == 'chisq':
            indep_test = chisq
        elif self.params['indep_test'] == 'gsq':
            indep_test = gsq
        elif self.params['indep_test'] == 'kci':
            indep_test = kci
        else:
            indep_test = fisherz
        
        # Run FCI algorithm
        G, edges = fci(
            X,
            independence_test_method=indep_test,
            alpha=self.params['alpha'],
            stable=self.params['stable'],
            max_path_length=self.params['max_path_length'],
            show_progress=False
        )
        
        # Convert to NetworkX graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        # Extract directed edges from PAG
        adj_matrix = G.graph
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
            'features': node_names,
            'note': 'FCI returns a PAG (Partial Ancestral Graph) which may include bidirected edges'
        }
        
        return self.learned_graph
