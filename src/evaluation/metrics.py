"""
Conditional independence metrics for evaluation.

Implements statistical tests for conditional independence.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Set
import networkx as nx


class ConditionalIndependenceTest:
    """Statistical tests for conditional independence."""
    
    def __init__(self, method: str = 'fisherz', alpha: float = 0.05):
        """
        Initialize CI test.
        
        Args:
            method: Test method ('fisherz', 'chisq', 'gsq')
            alpha: Significance level
        """
        self.method = method
        self.alpha = alpha
    
    def test(self, data: pd.DataFrame, x: str, y: str, z: List[str] = None) -> Tuple[bool, float]:
        """
        Test conditional independence X _|_ Y | Z.
        
        Args:
            data: Dataset
            x: First variable name
            y: Second variable name
            z: Conditioning set (list of variable names)
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        if z is None or len(z) == 0:
            # Marginal independence test
            return self._marginal_test(data, x, y)
        else:
            # Conditional independence test
            return self._conditional_test(data, x, y, z)
    
    def _marginal_test(self, data: pd.DataFrame, x: str, y: str) -> Tuple[bool, float]:
        """
        Test marginal independence X _|_ Y.
        
        Args:
            data: Dataset
            x: First variable
            y: Second variable
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        X = data[x].values
        Y = data[y].values
        
        if self.method == 'fisherz':
            # Pearson correlation test for continuous variables
            corr, p_value = stats.pearsonr(X, Y)
        elif self.method in ['chisq', 'gsq']:
            # Chi-square test for categorical variables
            contingency_table = pd.crosstab(data[x], data[y])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        else:
            # Default to correlation
            corr, p_value = stats.pearsonr(X, Y)
        
        is_independent = p_value > self.alpha
        
        return is_independent, p_value
    
    def _conditional_test(self, data: pd.DataFrame, x: str, y: str, z: List[str]) -> Tuple[bool, float]:
        """
        Test conditional independence X _|_ Y | Z.
        
        Uses partial correlation for continuous variables.
        
        Args:
            data: Dataset
            x: First variable
            y: Second variable
            z: Conditioning variables
            
        Returns:
            Tuple of (is_independent, p_value)
        """
        if self.method == 'fisherz':
            # Partial correlation test
            r_xy_z, p_value = self._partial_correlation_test(data, x, y, z)
        else:
            # For categorical, use conditional chi-square
            r_xy_z, p_value = self._partial_correlation_test(data, x, y, z)
        
        is_independent = p_value > self.alpha
        
        return is_independent, p_value
    
    def _partial_correlation_test(self, data: pd.DataFrame, x: str, y: str, 
                                  z: List[str]) -> Tuple[float, float]:
        """
        Compute partial correlation and test significance.
        
        Args:
            data: Dataset
            x: First variable
            y: Second variable
            z: Conditioning variables
            
        Returns:
            Tuple of (partial_correlation, p_value)
        """
        variables = [x, y] + z
        data_subset = data[variables].values
        
        # Remove rows with NaN
        data_subset = data_subset[~np.isnan(data_subset).any(axis=1)]
        
        if len(data_subset) < len(variables) + 1:
            # Not enough data
            return 0.0, 1.0
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data_subset.T)
        
        # Indices
        idx_x = 0
        idx_y = 1
        idx_z = list(range(2, len(variables)))
        
        if len(idx_z) == 0:
            # No conditioning variables, return marginal correlation
            r = corr_matrix[idx_x, idx_y]
        else:
            # Compute partial correlation using precision matrix
            try:
                prec_matrix = np.linalg.inv(corr_matrix)
                r = -prec_matrix[idx_x, idx_y] / np.sqrt(prec_matrix[idx_x, idx_x] * prec_matrix[idx_y, idx_y])
            except:
                # Singular matrix, return 0
                r = 0.0
        
        # Fisher's z-transformation
        n = len(data_subset)
        k = len(idx_z)
        
        if abs(r) >= 1:
            p_value = 0.0 if abs(r) > 1 else 1.0
        else:
            z = 0.5 * np.log((1 + r) / (1 - r))
            z_stat = z * np.sqrt(n - k - 3)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return r, p_value


def extract_independence_implications(G: nx.DiGraph) -> List[Tuple[str, str, Set[str]]]:
    """
    Extract conditional independence implications from a causal graph.
    
    Uses d-separation to find conditional independencies.
    
    Args:
        G: Causal graph
        
    Returns:
        List of (X, Y, Z) tuples representing X _|_ Y | Z
    """
    from itertools import combinations
    
    nodes = list(G.nodes())
    independencies = []
    
    # Check all pairs of nodes
    for x, y in combinations(nodes, 2):
        # Try different conditioning sets
        other_nodes = [n for n in nodes if n not in [x, y]]
        
        # Test with empty conditioning set
        if nx.d_separated(G, {x}, {y}, set()):
            independencies.append((x, y, set()))
        
        # Test with subsets of other nodes as conditioning sets
        for r in range(1, min(len(other_nodes) + 1, 4)):  # Limit to size 3 for efficiency
            for z_nodes in combinations(other_nodes, r):
                z_set = set(z_nodes)
                if nx.d_separated(G, {x}, {y}, z_set):
                    independencies.append((x, y, z_set))
    
    return independencies


def compute_ci_violation_score(data: pd.DataFrame, G: nx.DiGraph, 
                               method: str = 'fisherz', alpha: float = 0.05) -> float:
    """
    Compute conditional independence violation score.
    
    This measures how well the data satisfies the CI constraints
    implied by the causal graph.
    
    Args:
        data: Dataset
        G: Causal graph
        method: CI test method
        alpha: Significance level
        
    Returns:
        Violation score (lower is better)
    """
    ci_test = ConditionalIndependenceTest(method=method, alpha=alpha)
    
    # Extract independence implications from graph
    independencies = extract_independence_implications(G)
    
    if len(independencies) == 0:
        return 0.0
    
    # Test each independence
    violations = 0
    total_tests = 0
    
    for x, y, z in independencies:
        z_list = list(z) if z else []
        
        # Skip if variables not in data
        if x not in data.columns or y not in data.columns:
            continue
        if any(var not in data.columns for var in z_list):
            continue
        
        is_independent, p_value = ci_test.test(data, x, y, z_list)
        
        # If the graph says they should be independent but they're not, it's a violation
        if not is_independent:
            violations += 1
        
        total_tests += 1
    
    if total_tests == 0:
        return 0.0
    
    # Violation rate
    violation_score = violations / total_tests
    
    return violation_score
