import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple
import json
import os

from .metrics import ConditionalIndependenceTest, extract_independence_implications, compute_ci_violation_score


class CauTabBenchEvaluator:

    def __init__(self, method: str = 'fisherz', alpha: float = 0.05):

        self.method = method
        self.alpha = alpha
        self.ci_test = ConditionalIndependenceTest(method=method, alpha=alpha)
    
    def evaluate(self, real_data: pd.DataFrame, 
                synthetic_data: pd.DataFrame,
                causal_graph: nx.DiGraph) -> Dict[str, Any]:

        independencies = extract_independence_implications(causal_graph)
        
        real_results = self._test_independencies(real_data, independencies)
        synthetic_results = self._test_independencies(synthetic_data, independencies)
        
        agreement = self._compute_agreement(real_results, synthetic_results)
        
        real_violation = compute_ci_violation_score(real_data, causal_graph, self.method, self.alpha)
        synthetic_violation = compute_ci_violation_score(synthetic_data, causal_graph, self.method, self.alpha)
        
        quality_score = self._compute_quality_score(agreement, real_violation, synthetic_violation)
        
        return {
            'n_tests': len(independencies),
            'agreement_rate': agreement['agreement_rate'],
            'real_violation_rate': real_violation,
            'synthetic_violation_rate': synthetic_violation,
            'quality_score': quality_score,
            'detailed_agreement': agreement,
            'method': self.method,
            'alpha': self.alpha
        }
    
    def _test_independencies(self, data: pd.DataFrame, 
                            independencies: List[Tuple[str, str, set]]) -> List[Dict]:

        results = []
        
        for x, y, z in independencies:
            z_list = list(z) if z else []
            
            if x not in data.columns or y not in data.columns:
                continue
            if any(var not in data.columns for var in z_list):
                continue
            
            is_independent, p_value = self.ci_test.test(data, x, y, z_list)
            
            results.append({
                'x': x,
                'y': y,
                'z': z_list,
                'is_independent': is_independent,
                'p_value': p_value
            })
        
        return results
    
    def _compute_agreement(self, real_results: List[Dict], 
                          synthetic_results: List[Dict]) -> Dict[str, Any]:

        if len(real_results) != len(synthetic_results):
            raise ValueError("Real and synthetic results must have same length")
        
        n_tests = len(real_results)
        agreements = 0
        disagreements = []
        
        for real_res, syn_res in zip(real_results, synthetic_results):
            if real_res['is_independent'] == syn_res['is_independent']:
                agreements += 1
            else:
                disagreements.append({
                    'x': real_res['x'],
                    'y': real_res['y'],
                    'z': real_res['z'],
                    'real_independent': real_res['is_independent'],
                    'synthetic_independent': syn_res['is_independent']
                })
        
        agreement_rate = agreements / n_tests if n_tests > 0 else 0.0
        
        return {
            'agreement_rate': agreement_rate,
            'n_agreements': agreements,
            'n_disagreements': len(disagreements),
            'disagreements': disagreements
        }
    
    def _compute_quality_score(self, agreement: Dict, 
                              real_violation: float,
                              synthetic_violation: float) -> float:

        w_agreement = 0.6
        w_violation = 0.4
        
        agreement_score = agreement['agreement_rate']
        
        violation_diff = abs(synthetic_violation - real_violation)
        violation_score = 1.0 - violation_diff
        
        quality_score = w_agreement * agreement_score + w_violation * violation_score
        
        return quality_score
    
    def compare_models(self, real_data: pd.DataFrame,
                      synthetic_datasets: Dict[str, pd.DataFrame],
                      causal_graph: nx.DiGraph) -> Dict[str, Any]:

        results = {}
        
        for model_name, synthetic_data in synthetic_datasets.items():
            results[model_name] = self.evaluate(real_data, synthetic_data, causal_graph)
        
        rankings = sorted(results.items(), key=lambda x: x[1]['quality_score'], reverse=True)
        
        return {
            'model_results': results,
            'rankings': [(name, score['quality_score']) for name, score in rankings],
            'best_model': rankings[0][0] if rankings else None
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        results_serializable = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    def _make_serializable(self, obj):

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
