import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Set
import networkx as nx


class ConditionalIndependenceTest:

    def __init__(self, method: str = 'fisherz', alpha: float = 0.05):

        self.method = method
        self.alpha = alpha
    
    def test(self, data: pd.DataFrame, x: str, y: str, z: List[str] = None) -> Tuple[bool, float]:

        if z is None or len(z) == 0:
            return self._marginal_test(data, x, y)
        else:
            return self._conditional_test(data, x, y, z)
    
    def _marginal_test(self, data: pd.DataFrame, x: str, y: str) -> Tuple[bool, float]:

        X = data[x].values
        Y = data[y].values
        
        if self.method == 'fisherz':
            corr, p_value = stats.pearsonr(X, Y)
        elif self.method in ['chisq', 'gsq']:
            contingency_table = pd.crosstab(data[x], data[y])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        else:
            corr, p_value = stats.pearsonr(X, Y)
        
        is_independent = p_value > self.alpha
        
        return is_independent, p_value
    
    def _conditional_test(self, data: pd.DataFrame, x: str, y: str, z: List[str]) -> Tuple[bool, float]:

        if self.method == 'fisherz':
            r_xy_z, p_value = self._partial_correlation_test(data, x, y, z)
        else:
            r_xy_z, p_value = self._partial_correlation_test(data, x, y, z)
        
        is_independent = p_value > self.alpha
        
        return is_independent, p_value
    
    def _partial_correlation_test(self, data: pd.DataFrame, x: str, y: str, 
                                  z: List[str]) -> Tuple[float, float]:
        variables = [x, y] + z
        data_subset = data[variables].values
        
        data_subset = data_subset[~np.isnan(data_subset).any(axis=1)]
        
        if len(data_subset) < len(variables) + 1:
            return 0.0, 1.0
        
        corr_matrix = np.corrcoef(data_subset.T)
        
        idx_x = 0
        idx_y = 1
        idx_z = list(range(2, len(variables)))
        
        if len(idx_z) == 0:
            r = corr_matrix[idx_x, idx_y]
        else:
            try:
                prec_matrix = np.linalg.inv(corr_matrix)
                r = -prec_matrix[idx_x, idx_y] / np.sqrt(prec_matrix[idx_x, idx_x] * prec_matrix[idx_y, idx_y])
            except:
                r = 0.0
        
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
    from itertools import combinations
    
    nodes = list(G.nodes())
    independencies = []
    
    for x, y in combinations(nodes, 2):
        other_nodes = [n for n in nodes if n not in [x, y]]
        
        if nx.d_separated(G, {x}, {y}, set()):
            independencies.append((x, y, set()))
        
        for r in range(1, min(len(other_nodes) + 1, 4)):
            for z_nodes in combinations(other_nodes, r):
                z_set = set(z_nodes)
                if nx.d_separated(G, {x}, {y}, z_set):
                    independencies.append((x, y, z_set))
    
    return independencies


def compute_ci_violation_score(data: pd.DataFrame, G: nx.DiGraph, 
                               method: str = 'fisherz', alpha: float = 0.05) -> float:
    ci_test = ConditionalIndependenceTest(method=method, alpha=alpha)
    
    independencies = extract_independence_implications(G)
    
    if len(independencies) == 0:
        return 0.0
    
    violations = 0
    total_tests = 0
    
    for x, y, z in independencies:
        z_list = list(z) if z else []
        
        if x not in data.columns or y not in data.columns:
            continue
        if any(var not in data.columns for var in z_list):
            continue
        
        is_independent, p_value = ci_test.test(data, x, y, z_list)
        
        if not is_independent:
            violations += 1
        
        total_tests += 1
    
    if total_tests == 0:
        return 0.0
    
    violation_score = violations / total_tests
    
    return violation_score
