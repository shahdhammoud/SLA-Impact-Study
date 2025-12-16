"""
NOTEARS algorithm implementation.

NOTEARS: Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning.
Uses continuous optimization to learn DAG structure.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from .base import BaseStructureLearner


class NOTEARSLearner(BaseStructureLearner):
    """NOTEARS algorithm for causal structure learning."""
    
    def __init__(self, **kwargs):
        """
        Initialize NOTEARS learner.
        
        Args:
            **kwargs: NOTEARS algorithm parameters
        """
        super().__init__(algorithm_name="NOTEARS", **kwargs)
        
        # Default parameters
        default_params = {
            'lambda1': 0.1,  # L1 regularization
            'lambda2': 0.1,  # L2 regularization
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.3,
            'loss_type': 'l2',
            'use_gpu': True
        }
        
        # Update with provided params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() and self.params['use_gpu'] else 'cpu'
    
    def fit(self, data: pd.DataFrame, **kwargs) -> nx.DiGraph:
        """
        Learn causal structure using NOTEARS algorithm.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Learned causal graph
        """
        # Convert to numpy array and normalize
        X = data.values
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        node_names = list(data.columns)
        
        # Run NOTEARS optimization
        W_est = self._notears_linear(X)
        
        # Threshold to get final adjacency matrix
        W_est[np.abs(W_est) < self.params['w_threshold']] = 0
        
        # Convert to NetworkX graph
        self.learned_graph = nx.DiGraph()
        self.learned_graph.add_nodes_from(node_names)
        
        n_nodes = len(node_names)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if W_est[i, j] != 0:
                    self.learned_graph.add_edge(node_names[i], node_names[j])
        
        self.is_fitted = True
        
        # Store metadata
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(node_names),
            'n_edges': self.learned_graph.number_of_edges(),
            'features': node_names,
            'weight_matrix_sparsity': float(np.mean(W_est == 0))
        }
        
        return self.learned_graph
    
    def _notears_linear(self, X: np.ndarray) -> np.ndarray:
        """
        NOTEARS algorithm for linear SEM.
        
        Args:
            X: Data matrix [n_samples, n_features]
            
        Returns:
            Estimated weighted adjacency matrix
        """
        n, d = X.shape
        
        # Convert to torch tensor
        X_torch = torch.FloatTensor(X).to(self.device)
        
        # Initialize weight matrix
        W = torch.zeros(d, d, requires_grad=True, device=self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam([W], lr=0.01)
        
        # Augmented Lagrangian parameters
        rho = 1.0
        alpha = 0.0
        h = np.inf
        
        for iter in range(self.params['max_iter']):
            optimizer.zero_grad()
            
            # Compute loss
            loss = self._compute_loss(X_torch, W)
            
            # Compute DAG constraint h(W)
            h_val = self._compute_h(W)
            
            # Augmented Lagrangian
            obj = loss + 0.5 * rho * h_val * h_val + alpha * h_val
            
            obj.backward()
            optimizer.step()
            
            # Update augmented Lagrangian parameters
            h = h_val.item()
            if h > 0.25 * self.params['h_tol']:
                alpha += rho * h
                rho *= 10
            
            if h <= self.params['h_tol']:
                break
        
        return W.detach().cpu().numpy()
    
    def _compute_loss(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for NOTEARS.
        
        Args:
            X: Data tensor
            W: Weight matrix
            
        Returns:
            Loss value
        """
        n, d = X.shape
        
        # Least squares loss
        if self.params['loss_type'] == 'l2':
            loss = 0.5 / n * torch.sum((X - X @ W) ** 2)
        else:
            loss = 0.5 / n * torch.sum((X - X @ W) ** 2)
        
        # L1 regularization
        loss += self.params['lambda1'] * torch.sum(torch.abs(W))
        
        # L2 regularization
        loss += 0.5 * self.params['lambda2'] * torch.sum(W ** 2)
        
        return loss
    
    def _compute_h(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute DAG constraint h(W) = tr(e^(W*W)) - d.
        
        Args:
            W: Weight matrix
            
        Returns:
            DAG constraint value
        """
        d = W.shape[0]
        M = torch.eye(d, device=W.device) + W * W / d
        return torch.trace(torch.matrix_power(M, d)) - d
