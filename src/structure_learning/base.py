from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional
import os
import json


class BaseStructureLearner(ABC):

    def __init__(self, algorithm_name: str, **kwargs):

        self.algorithm_name = algorithm_name
        self.params = kwargs
        self.learned_graph = None
        self.is_fitted = False
        self.metadata = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure from data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Learned causal graph
        """
        pass
    
    def get_graph(self) -> nx.DiGraph:

        if not self.is_fitted:
            raise ValueError("Structure must be learned before accessing graph")
        
        return self.learned_graph
    
    def get_adjacency_matrix(self) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Structure must be learned before accessing adjacency matrix")
        
        nodes = sorted(list(self.learned_graph.nodes()))
        n_nodes = len(nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        for source, target in self.learned_graph.edges():
            i = node_to_idx[source]
            j = node_to_idx[target]
            adj_matrix[i, j] = 1
        
        return adj_matrix
    
    def save(self, filepath: str):

        if not self.is_fitted:
            raise ValueError("Structure must be learned before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        edges = list(self.learned_graph.edges())
        nodes = list(self.learned_graph.nodes())
        
        save_dict = {
            'algorithm': self.algorithm_name,
            'params': self.params,
            'nodes': nodes,
            'edges': edges,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    def load(self, filepath: str):

        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        self.algorithm_name = save_dict['algorithm']
        self.params = save_dict['params']
        self.metadata = save_dict['metadata']
        
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(save_dict['nodes'])
        self.learned_graph.add_edges_from(save_dict['edges'])
        
        self.is_fitted = True
    
    def get_params(self) -> Dict[str, Any]:

        return self.params.copy()
    
    def set_params(self, **params):

        self.params.update(params)
