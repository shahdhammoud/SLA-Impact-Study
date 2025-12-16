"""
PC (Peter-Clark) algorithm implementation using causal-learn.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci

from .base import BaseStructureLearner


class PCLearner(BaseStructureLearner):
    """PC algorithm for causal structure learning."""
    
    def __init__(self, **kwargs):
        """
        Initialize PC learner.
        
        Args:
            **kwargs: PC algorithm parameters
        """
        super().__init__(algorithm_name="PC", **kwargs)
        
        # Default parameters
        default_params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'stable': True,
            'max_cond_vars': -1
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure using PC algorithm.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Learned causal graph
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
        
        # Run PC algorithm
        cg = pc(
            X,
            alpha=self.params['alpha'],
            indep_test=indep_test,
            stable=self.params['stable'],
            max_cond_vars=self.params['max_cond_vars'],
            show_progress=False
        )
        
        # Convert to NetworkX graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        # Extract edges from causal graph
        adj_matrix = cg.G.graph
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
