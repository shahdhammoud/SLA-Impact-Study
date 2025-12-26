"""
CDNOD (Causal Discovery from Nonstationary/heterogeneous Data) algorithm implementation.

CDNOD is designed to handle nonstationary or heterogeneous data and works well
with discrete/categorical variables.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

from .base import BaseStructureLearner


class CDNODLearner(BaseStructureLearner):
    """CDNOD algorithm for causal structure learning from heterogeneous data."""

    def __init__(self, **kwargs):
        """
        Initialize CDNOD learner.

        Args:
            **kwargs: CDNOD algorithm parameters
        """
        super().__init__(algorithm_name="CDNOD", **kwargs)

        # Default parameters
        default_params = {
            'alpha': 0.05,
            'indep_test': 'chisq',  # Better for discrete data
            'stable': True,
            'max_cond_vars': 4
        }

        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure using CDNOD algorithm.

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Learned causal graph
        """
        try:
            from causallearn.search.ConstraintBased.CDNOD import cdnod
            from causallearn.utils.cit import chisq, fisherz, gsq, kci
        except ImportError:
            # Fallback to PC-based approach if CDNOD not available
            print("CDNOD not available in causal-learn, using PC with appropriate test")
            return self._fallback_pc(data)

        # Convert to numpy array
        X = data.values
        node_names = list(data.columns)

        # Create a simple time/domain index (required by CDNOD)
        # CDNOD expects c_indx as a 2D array with shape (n_samples, 1)
        c_indx = np.zeros((len(X), 1), dtype=int)

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
            indep_test = chisq  # Default to chi-square for discrete

        try:
            # Run CDNOD algorithm
            cg = cdnod(
                X,
                c_indx,
                alpha=self.params['alpha'],
                indep_test=indep_test,
                stable=self.params['stable']
            )

            # Convert to NetworkX graph
            self.learned_graph = nx.DiGraph()
            self.learned_graph.add_nodes_from(node_names)

            # Extract edges from the learned graph
            adj_matrix = cg.G.graph
            n_nodes = len(node_names)

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                        # Directed edge i -> j
                        self.learned_graph.add_edge(node_names[i], node_names[j])
                    elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
                        # Directed edge j -> i
                        self.learned_graph.add_edge(node_names[j], node_names[i])
                    elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                        # Undirected edge - add both directions
                        self.learned_graph.add_edge(node_names[i], node_names[j])

        except Exception as e:
            print(f"CDNOD failed: {e}, using fallback PC")
            return self._fallback_pc(data)

        self.is_fitted = True

        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names,
            'algorithm': 'CDNOD',
            'note': 'CDNOD handles nonstationary/heterogeneous data'
        }

        return self.learned_graph

    def _fallback_pc(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Fallback to PC algorithm with chi-square test if CDNOD fails.

        Args:
            data: Input data

        Returns:
            Learned causal graph
        """
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import chisq, gsq

        X = data.values
        node_names = list(data.columns)

        # Use chi-square test for discrete data
        indep_test = chisq if self.params['indep_test'] == 'chisq' else gsq

        cg = pc(
            X,
            alpha=self.params['alpha'],
            indep_test=indep_test,
            stable=self.params['stable'],
            show_progress=False
        )

        # Convert to NetworkX graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)

        adj_matrix = cg.G.graph
        n_nodes = len(node_names)

        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    self.learned_graph.add_edge(node_names[i], node_names[j])
                elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
                    self.learned_graph.add_edge(node_names[j], node_names[i])
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                    self.learned_graph.add_edge(node_names[i], node_names[j])

        self.is_fitted = True

        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names,
            'algorithm': 'PC (CDNOD fallback)',
            'note': 'Used PC with chi-square test as CDNOD fallback'
        }

        return self.learned_graph

