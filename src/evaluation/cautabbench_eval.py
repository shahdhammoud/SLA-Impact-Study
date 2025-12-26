"""
CauTabBench-style evaluation methodology.

Evaluates generative models by comparing how well real and synthetic data
satisfy conditional independence constraints implied by a causal structure.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple
import json
import os

from .metrics import ConditionalIndependenceTest, extract_independence_implications, compute_ci_violation_score


class CauTabBenchEvaluator:
    """
    Evaluate generative models using CauTabBench methodology.
    
    Compares real and synthetic data against conditional independence
    constraints implied by a causal structure.
    """
    
    def __init__(self, method: str = 'fisherz', alpha: float = 0.05):
        """
        Initialize evaluator.
        
        Args:
            method: CI test method
            alpha: Significance level
        """
        self.method = method
        self.alpha = alpha
        self.ci_test = ConditionalIndependenceTest(method=method, alpha=alpha)
    
    def evaluate(self, real_data: pd.DataFrame, 
                synthetic_data: pd.DataFrame,
                causal_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Evaluate synthetic data quality using CauTabBench methodology.
        
        Args:
            real_data: Real dataset
            synthetic_data: Synthetic dataset
            causal_graph: Causal structure graph
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract independence constraints from graph
        independencies = extract_independence_implications(causal_graph)
        
        # Test independencies on both datasets
        real_results = self._test_independencies(real_data, independencies)
        synthetic_results = self._test_independencies(synthetic_data, independencies)
        
        # Compute agreement metrics
        agreement = self._compute_agreement(real_results, synthetic_results)
        
        # Compute violation scores
        real_violation = compute_ci_violation_score(real_data, causal_graph, self.method, self.alpha)
        synthetic_violation = compute_ci_violation_score(synthetic_data, causal_graph, self.method, self.alpha)
        
        # Compute overall quality score
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
        """
        Test all independence constraints on data.
        
        Args:
            data: Dataset
            independencies: List of (X, Y, Z) tuples
            
        Returns:
            List of test results
        """
        results = []
        
        for x, y, z in independencies:
            z_list = list(z) if z else []
            
            # Skip if variables not in data
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
        """
        Compute agreement between real and synthetic results.
        
        Args:
            real_results: Test results on real data
            synthetic_results: Test results on synthetic data
            
        Returns:
            Agreement metrics
        """
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
        """
        Compute overall quality score.
        
        Higher score means better synthetic data quality.
        
        Args:
            agreement: Agreement metrics
            real_violation: Real data violation rate
            synthetic_violation: Synthetic data violation rate
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Weight factors
        w_agreement = 0.6
        w_violation = 0.4
        
        # Agreement score (higher is better)
        agreement_score = agreement['agreement_rate']
        
        # Violation score (lower violation is better)
        # We want synthetic violation to be similar to real violation
        violation_diff = abs(synthetic_violation - real_violation)
        violation_score = 1.0 - violation_diff
        
        # Combined score
        quality_score = w_agreement * agreement_score + w_violation * violation_score
        
        return quality_score
    
    def compare_models(self, real_data: pd.DataFrame,
                      synthetic_datasets: Dict[str, pd.DataFrame],
                      causal_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Compare multiple generative models.
        
        Args:
            real_data: Real dataset
            synthetic_datasets: Dictionary mapping model names to synthetic datasets
            causal_graph: Causal structure graph
            
        Returns:
            Comparison results with rankings
        """
        results = {}
        
        for model_name, synthetic_data in synthetic_datasets.items():
            results[model_name] = self.evaluate(real_data, synthetic_data, causal_graph)
        
        # Rank models by quality score
        rankings = sorted(results.items(), key=lambda x: x[1]['quality_score'], reverse=True)
        
        return {
            'model_results': results,
            'rankings': [(name, score['quality_score']) for name, score in rankings],
            'best_model': rankings[0][0] if rankings else None
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        results_serializable = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
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
