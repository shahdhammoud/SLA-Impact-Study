import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional

import lingam

from .base import BaseStructureLearner


class LiNGAMLearner(BaseStructureLearner):
    def __init__(self, **kwargs):
        super().__init__(algorithm_name="LiNGAM", **kwargs)
        
        default_params = {
            'measure': 'pwling',
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        X = data.values
        node_names = list(data.columns)
        
        if self.params['measure'] == 'pwling':
            model = lingam.DirectLiNGAM(random_state=self.params['random_state'])
        else:
            model = lingam.DirectLiNGAM(random_state=self.params['random_state'])
        
        model.fit(X)
        
        adj_matrix = model.adjacency_matrix_
        
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        n_nodes = len(node_names)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:
                    self.learned_graph.add_edge(node_names[i], node_names[j])
        
        self.is_fitted = True
        
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names,
            'causal_order': [node_names[i] for i in model.causal_order_],
            'note': 'LiNGAM assumes linear relationships and non-Gaussian noise'
        }
        
        return self.learned_graph
