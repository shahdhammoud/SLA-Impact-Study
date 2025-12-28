import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, gsq, kci

from .base import BaseStructureLearner


class FCILearner(BaseStructureLearner):
    def __init__(self, **kwargs):
        super().__init__(algorithm_name="FCI", **kwargs)
        
        default_params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'stable': True,
            'max_path_length': -1
        }
        
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        X = data.values
        node_names = list(data.columns)
        
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
        
        G, edges = fci(
            X,
            independence_test_method=indep_test,
            alpha=self.params['alpha'],
            stable=self.params['stable'],
            max_path_length=self.params['max_path_length'],
            show_progress=False
        )
        
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        adj_matrix = G.graph
        n_nodes = len(node_names)
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
                    self.learned_graph.add_edge(node_names[i], node_names[j])
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    self.learned_graph.add_edge(node_names[j], node_names[i])
        
        self.is_fitted = True
        
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names,
            'note': 'FCI returns a PAG (Partial Ancestral Graph) which may include bidirected edges'
        }
        
        return self.learned_graph
