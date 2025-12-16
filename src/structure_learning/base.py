"""
Base interface for structure learning algorithms.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional
import os
import json


class BaseStructureLearner(ABC):
    """Abstract base class for causal structure learning algorithms."""
    
    def __init__(self, algorithm_name: str, **kwargs):
        """
        Initialize structure learner.
        
        Args:
            algorithm_name: Name of the algorithm
            **kwargs: Algorithm-specific parameters
        """
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
        """
        Get learned causal graph.
        
        Returns:
            NetworkX DiGraph
        """
        if not self.is_fitted:
            raise ValueError("Structure must be learned before accessing graph")
        
        return self.learned_graph
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get adjacency matrix of learned graph.
        
        Returns:
            Adjacency matrix
        """
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
        """
        Save learned structure to file.
        
        Args:
            filepath: Path to save structure
        """
        if not self.is_fitted:
            raise ValueError("Structure must be learned before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as edge list
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
        """
        Load learned structure from file.
        
        Args:
            filepath: Path to load structure from
        """
        with open(filepath, 'r') as f:
            save_dict = json.load(f)
        
        self.algorithm_name = save_dict['algorithm']
        self.params = save_dict['params']
        self.metadata = save_dict['metadata']
        
        # Reconstruct graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(save_dict['nodes'])
        self.learned_graph.add_edges_from(save_dict['edges'])
        
        self.is_fitted = True
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.params.copy()
    
    def set_params(self, **params):
        """
        Set algorithm parameters.
        
        Args:
            **params: Parameters to set
        """
        self.params.update(params)
