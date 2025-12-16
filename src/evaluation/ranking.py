"""
Ranking comparison utilities.

Compare model rankings across different structure learning methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
from scipy.stats import kendalltau, spearmanr


class RankingComparator:
    """Compare model rankings across different causal structures."""
    
    def __init__(self):
        """Initialize ranking comparator."""
        self.comparisons = {}
    
    def add_ranking(self, structure_name: str, rankings: List[Tuple[str, float]]):
        """
        Add a ranking result.
        
        Args:
            structure_name: Name of the structure used (e.g., 'ground_truth', 'pc_learned')
            rankings: List of (model_name, score) tuples, sorted by score
        """
        self.comparisons[structure_name] = rankings
    
    def compare_rankings(self, baseline: str = 'ground_truth') -> Dict[str, Any]:
        """
        Compare all rankings against a baseline.
        
        Args:
            baseline: Name of baseline structure (usually 'ground_truth')
            
        Returns:
            Comparison results
        """
        if baseline not in self.comparisons:
            raise ValueError(f"Baseline '{baseline}' not found in comparisons")
        
        baseline_ranking = self.comparisons[baseline]
        results = {
            'baseline': baseline,
            'comparisons': {}
        }
        
        for struct_name, ranking in self.comparisons.items():
            if struct_name == baseline:
                continue
            
            comparison = self._compare_two_rankings(baseline_ranking, ranking)
            results['comparisons'][struct_name] = comparison
        
        return results
    
    def _compare_two_rankings(self, ranking1: List[Tuple[str, float]], 
                             ranking2: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Compare two rankings.
        
        Args:
            ranking1: First ranking
            ranking2: Second ranking
            
        Returns:
            Comparison metrics
        """
        # Get model names in order
        models1 = [name for name, _ in ranking1]
        models2 = [name for name, _ in ranking2]
        
        # Check if same models
        if set(models1) != set(models2):
            return {
                'error': 'Different sets of models',
                'models1': models1,
                'models2': models2
            }
        
        # Get scores
        scores1 = [score for _, score in ranking1]
        scores2_dict = {name: score for name, score in ranking2}
        scores2 = [scores2_dict[name] for name in models1]
        
        # Compute rank correlation
        kendall_tau, kendall_p = kendalltau(scores1, scores2)
        spearman_rho, spearman_p = spearmanr(scores1, scores2)
        
        # Compute positional differences
        rank1 = {name: i for i, (name, _) in enumerate(ranking1)}
        rank2 = {name: i for i, (name, _) in enumerate(ranking2)}
        
        positional_diffs = {name: abs(rank1[name] - rank2[name]) for name in models1}
        avg_positional_diff = np.mean(list(positional_diffs.values()))
        max_positional_diff = np.max(list(positional_diffs.values()))
        
        # Check if top model is same
        top1 = models1[0]
        top2 = models2[0]
        top_model_match = (top1 == top2)
        
        return {
            'kendall_tau': float(kendall_tau),
            'kendall_p_value': float(kendall_p),
            'spearman_rho': float(spearman_rho),
            'spearman_p_value': float(spearman_p),
            'avg_positional_difference': float(avg_positional_diff),
            'max_positional_difference': int(max_positional_diff),
            'top_model_match': top_model_match,
            'top_model_baseline': top1,
            'top_model_comparison': top2,
            'positional_differences': positional_diffs,
            'ranking1': ranking1,
            'ranking2': ranking2
        }
    
    def get_consensus_ranking(self) -> List[Tuple[str, float]]:
        """
        Compute consensus ranking across all structures.
        
        Uses average scores across all structures.
        
        Returns:
            Consensus ranking
        """
        if not self.comparisons:
            return []
        
        # Get all model names
        all_models = set()
        for rankings in self.comparisons.values():
            all_models.update([name for name, _ in rankings])
        
        # Compute average scores
        avg_scores = {}
        for model in all_models:
            scores = []
            for rankings in self.comparisons.values():
                for name, score in rankings:
                    if name == model:
                        scores.append(score)
            
            avg_scores[model] = np.mean(scores) if scores else 0.0
        
        # Sort by average score
        consensus = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        return consensus
    
    def summarize(self) -> Dict[str, Any]:
        """
        Generate summary of ranking comparisons.
        
        Returns:
            Summary dictionary
        """
        if not self.comparisons:
            return {'error': 'No comparisons available'}
        
        summary = {
            'n_structures': len(self.comparisons),
            'structures': list(self.comparisons.keys()),
            'n_models': len(self.comparisons[list(self.comparisons.keys())[0]]),
            'rankings': {}
        }
        
        # Add each ranking
        for struct_name, ranking in self.comparisons.items():
            summary['rankings'][struct_name] = {
                'order': [name for name, _ in ranking],
                'scores': [float(score) for _, score in ranking]
            }
        
        # Add consensus ranking
        consensus = self.get_consensus_ranking()
        summary['consensus'] = {
            'order': [name for name, _ in consensus],
            'scores': [float(score) for _, score in consensus]
        }
        
        # Compare all pairs
        if 'ground_truth' in self.comparisons:
            comparison_results = self.compare_rankings('ground_truth')
            summary['comparisons_vs_ground_truth'] = comparison_results['comparisons']
        
        return summary
    
    def save(self, filepath: str):
        """
        Save ranking comparisons to file.
        
        Args:
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        summary = self.summarize()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load ranking comparisons from file.
        
        Args:
            filepath: Path to load from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct comparisons
        self.comparisons = {}
        for struct_name, info in data['rankings'].items():
            rankings = list(zip(info['order'], info['scores']))
            self.comparisons[struct_name] = rankings
