"""
Ranking comparison utilities.

Evaluate causal discovery algorithms as synthetic data evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class RankingComparator:
    """Evaluate structural learning algorithms for synthetic data quality assessment."""

    def __init__(self):
        """Initialize ranking comparator."""
        self.ground_truths = {}  # {algorithm: G_true}
        self.real_structures = {}  # {algorithm: G_real}
        self.synthetic_structures = {}  # {model_name: {algorithm: G_synthetic}}
        self.reliability_scores = {}  # {algorithm: {model_name: score}}
        self.cautabench_scores = {}  # {model_name: score}

    def load_ground_truth(self, algorithm: str, graph_path: str):
        """
        Load ground truth causal graph from txt file.

        Args:
            algorithm: Causal discovery algorithm name (pc, ges, fci, etc.)
            graph_path: Path to ground truth txt file
        """
        graph = self._load_graph_from_txt(graph_path)
        self.ground_truths[algorithm] = graph
        print(f"✅ Loaded ground truth for {algorithm}")

    def load_real_structure(self, algorithm: str, structure_path: str):
        """
        Load G_real (structure learned from real data).

        Args:
            algorithm: Causal discovery algorithm name
            structure_path: Path to learned structure JSON file
        """
        graph = self._load_graph_from_json(structure_path)
        self.real_structures[algorithm] = graph
        print(f"✅ Loaded real structure for {algorithm}")

    def load_synthetic_structure(self, model_name: str, algorithm: str, structure_path: str):
        """
        Load G_synthetic (structure learned from synthetic data).

        Args:
            model_name: Name of generative model (gmm, ctgan, tabddpm)
            algorithm: Causal discovery algorithm name
            structure_path: Path to learned structure JSON file
        """
        if model_name not in self.synthetic_structures:
            self.synthetic_structures[model_name] = {}

        graph = self._load_graph_from_json(structure_path)
        self.synthetic_structures[model_name][algorithm] = graph
        print(f"✅ Loaded synthetic structure for {model_name}/{algorithm}")

    def compute_algorithm_reliability(self, model_name: str, algorithm: str) -> float:
        """
        Compute custom reliability score for a generative model and algorithm.

        Reliability Score = shd_synthetic_real / shd_synthetic_true
        Where:
            shd_synthetic_real: SHD between the graph learned from synthetic data and the graph learned from real data
            shd_synthetic_true: SHD between the graph learned from synthetic data and the ground truth graph
        If shd_synthetic_true == 0, reliability score is set to 1.0.
        Returns a float value (not percentage).
        """
        # Validate inputs
        if algorithm not in self.ground_truths:
            raise ValueError(f"Ground truth for {algorithm} not loaded")
        if algorithm not in self.real_structures:
            raise ValueError(f"Real structure for {algorithm} not loaded")
        if model_name not in self.synthetic_structures or \
           algorithm not in self.synthetic_structures[model_name]:
            raise ValueError(f"Synthetic structure for {model_name}/{algorithm} not loaded")

        # Get graphs
        g_true = self.ground_truths[algorithm]
        g_real = self.real_structures[algorithm]
        g_synthetic = self.synthetic_structures[model_name][algorithm]

        # Compute SHDs
        shd_synthetic_true = self._compute_shd(g_synthetic, g_true)
        shd_synthetic_real = self._compute_shd(g_synthetic, g_real)

        if shd_synthetic_true == 0:
            return 1.0
        return shd_synthetic_real / shd_synthetic_true

    def compute_all_reliability_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Compute reliability scores for all models and algorithms.

        Returns:
            Dictionary: {algorithm: {model_name: reliability_score}}
        """
        results = {}

        for algorithm in self.ground_truths.keys():
            if algorithm not in self.real_structures:
                continue

            results[algorithm] = {}

            for model_name in self.synthetic_structures.keys():
                if algorithm in self.synthetic_structures[model_name]:
                    score = self.compute_algorithm_reliability(model_name, algorithm)
                    results[algorithm][model_name] = score

        self.reliability_scores = results
        return results

    def rank_models_by_algorithm(self, algorithm: str) -> List[Tuple[str, float]]:
        """
        Rank models by reliability score for a specific algorithm.

        Args:
            algorithm: Causal discovery algorithm name

        Returns:
            List of (model_name, reliability_score) tuples, sorted by score (higher is better)
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        if algorithm not in self.reliability_scores:
            return []

        rankings = list(self.reliability_scores[algorithm].items())
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def rank_algorithms_by_reliability(self) -> List[Tuple[str, float]]:
        """
        Rank algorithms by average reliability across all models.

        Returns:
            List of (algorithm, avg_reliability) tuples, sorted by avg (higher is better)
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        avg_scores = {}
        for algorithm, model_scores in self.reliability_scores.items():
            if model_scores:
                avg_scores[algorithm] = np.mean(list(model_scores.values()))

        rankings = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        return rankings

    def add_cautabench_score(self, model_name: str, score: float):
        """
        Add CauTabBench quality score for a model.

        Args:
            model_name: Name of generative model
            score: Overall quality score from CauTabBench
        """
        self.cautabench_scores[model_name] = score

    def rank_by_cautabench(self) -> List[Tuple[str, float]]:
        """
        Rank models by CauTabBench score.

        Returns:
            List of (model_name, score) tuples, sorted by score (higher is better)
        """
        rankings = sorted(self.cautabench_scores.items(), key=lambda x: x[1], reverse=True)
        return rankings

    def summarize(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of algorithm reliability and model rankings.

        Returns:
            Summary dictionary
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        summary = {
            'models': list(self.synthetic_structures.keys()),
            'algorithms': list(self.ground_truths.keys()),
            'reliability_scores': self.reliability_scores,
            'rankings': {}
        }

        # Rankings by algorithm (which model has best structure)
        for algorithm in self.ground_truths.keys():
            ranking = self.rank_models_by_algorithm(algorithm)
            if ranking:
                summary['rankings'][f'{algorithm}_model_ranking'] = {
                    'order': [name for name, _ in ranking],
                    'scores': [float(score) for _, score in ranking]
                }

        # Algorithm reliability ranking (which algorithm is most reliable)
        algo_ranking = self.rank_algorithms_by_reliability()
        if algo_ranking:
            summary['rankings']['algorithm_reliability_ranking'] = {
                'order': [name for name, _ in algo_ranking],
                'avg_scores': [float(score) for _, score in algo_ranking]
            }

        # CauTabBench ranking if available
        if self.cautabench_scores:
            cautabench_ranking = self.rank_by_cautabench()
            summary['rankings']['cautabench_ranking'] = {
                'order': [name for name, _ in cautabench_ranking],
                'scores': [float(score) for _, score in cautabench_ranking]
            }

        # Overall model ranking (average reliability across all algorithms)
        overall_scores = {}
        for model_name in self.synthetic_structures.keys():
            scores = []
            for algorithm in self.ground_truths.keys():
                if algorithm in self.reliability_scores and \
                   model_name in self.reliability_scores[algorithm]:
                    scores.append(self.reliability_scores[algorithm][model_name])
            if scores:
                overall_scores[model_name] = np.mean(scores)

        if overall_scores:
            overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            summary['rankings']['overall_reliability_ranking'] = {
                'order': [name for name, _ in overall_ranking],
                'avg_scores': [float(score) for _, score in overall_ranking]
            }

        return summary

    def _load_graph_from_txt(self, filepath: str) -> nx.DiGraph:
        """Load causal graph from ground truth txt file."""
        graph = nx.DiGraph()

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle different formats: "A B" or "A -> B" or "A->B"
                    if '->' in line:
                        parts = line.split('->')
                    elif '\t' in line:
                        parts = line.split('\t')
                    else:
                        parts = line.split()

                    if len(parts) >= 2:
                        source = parts[0].strip()
                        target = parts[1].strip()
                        graph.add_edge(source, target)

        return graph

    def _load_graph_from_json(self, filepath: str) -> nx.DiGraph:
        """Load causal graph from JSON structure file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        graph = nx.DiGraph()

        # Add nodes
        for node in data.get('nodes', []):
            graph.add_node(node)

        # Add edges
        for edge in data.get('edges', []):
            if isinstance(edge, dict):
                graph.add_edge(edge.get('from') or edge.get('source'),
                             edge.get('to') or edge.get('target'))
            elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                graph.add_edge(edge[0], edge[1])

        return graph

    def _compute_shd(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> int:
        """
        Compute Structural Hamming Distance between two graphs.

        SHD counts: missing edges + extra edges + reversed edges

        Args:
            graph1: First graph
            graph2: Second graph (reference)

        Returns:
            SHD value
        """
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())

        # Missing edges (in graph2 but not in graph1)
        missing = len(edges2 - edges1)

        # Extra edges (in graph1 but not in graph2)
        extra = len(edges1 - edges2)

        # Reversed edges
        reversed_edges = 0
        for u, v in edges1:
            if (v, u) in edges2 and (u, v) not in edges2:
                reversed_edges += 1

        shd = missing + extra - reversed_edges

        return shd

    def _compute_graph_metrics(self, graph_pred: nx.DiGraph, graph_true: nx.DiGraph) -> dict:
        """
        Compute SHD, Precision, Recall, F1 Score for predicted vs. true graph.
        Returns a dictionary with all metrics.
        """
        pred_edges = set(graph_pred.edges())
        true_edges = set(graph_true.edges())
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = self._compute_shd(graph_pred, graph_true)
        return {"SHD": shd, "Precision": precision, "Recall": recall, "F1": f1}

    def compute_all_metrics(self) -> dict:
        """
        Compute SHD, Precision, Recall, F1 for all (model, algorithm) pairs.
        Returns a nested dictionary: metrics[algorithm][model] = {...}
        """
        metrics = {}
        for algorithm in self.ground_truths.keys():
            metrics[algorithm] = {}
            for model_name in self.synthetic_structures.keys():
                if algorithm in self.synthetic_structures[model_name]:
                    g_true = self.ground_truths[algorithm]
                    g_pred = self.synthetic_structures[model_name][algorithm]
                    metrics[algorithm][model_name] = self._compute_graph_metrics(g_pred, g_true)
        self.all_metrics = metrics
        return metrics

    def plot_metric_vs_reliability(self, metric_name: str, output_dir: str, dataset_name: str):
        """
        Plot each metric (F1, SHD, Precision, Recall, Composite) vs reliability score for each algorithm.
        """
        if not hasattr(self, "all_metrics"):
            self.compute_all_metrics()
        if not self.reliability_scores:
            self.compute_all_reliability_scores()
        for algorithm in self.ground_truths.keys():
            models = []
            metric_vals = []
            reliability_vals = []
            for model in self.synthetic_structures.keys():
                if model in self.all_metrics[algorithm] and model in self.reliability_scores[algorithm]:
                    models.append(model)
                    if metric_name == "Composite":
                        m = self.all_metrics[algorithm][model]
                        metric_vals.append((m["F1"] + m["Precision"] + m["Recall"]) / 3.0)
                    else:
                        metric_vals.append(self.all_metrics[algorithm][model][metric_name])
                    reliability_vals.append(self.reliability_scores[algorithm][model])
            if not models:
                continue
            # Normalize metric if needed
            if metric_name == "SHD":
                max_shd = max(metric_vals) if metric_vals else 1
                metric_vals = [v / max_shd for v in metric_vals]
            # Plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(reliability_vals, metric_vals, s=200, edgecolors='black', alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model, (reliability_vals[i], metric_vals[i]), fontsize=12, fontweight='bold', xytext=(10, 10), textcoords='offset points')
            ax.set_xlabel(f'Reliability Score ({algorithm})', fontsize=13, fontweight='bold')
            ax.set_ylabel(f'{metric_name} (normalized)', fontsize=13, fontweight='bold')
            ax.set_title(f'{metric_name} vs Reliability ({algorithm})', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{dataset_name}_{metric_name.lower()}_vs_reliability_{algorithm.lower()}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_regression_metric_vs_reliability(self, metric_name: str, output_dir: str, dataset_name: str):
        """
        Plot regression metric (R2) vs reliability score for each algorithm.
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()
        for algorithm in self.ground_truths.keys():
            models = []
            metric_vals = []
            reliability_vals = []
            for model in self.synthetic_structures.keys():
                # Look for regression metrics JSON file
                metrics_path = os.path.join('outputs', 'evaluations', f'{dataset_name}_{model}_regression_metrics.json')
                if os.path.exists(metrics_path) and model in self.reliability_scores[algorithm]:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    if metric_name in metrics:
                        models.append(model)
                        metric_vals.append(metrics[metric_name])
                        reliability_vals.append(self.reliability_scores[algorithm][model])
            if not models:
                continue
            # Plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(reliability_vals, metric_vals, s=200, edgecolors='black', alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model, (reliability_vals[i], metric_vals[i]), fontsize=12, fontweight='bold', xytext=(10, 10), textcoords='offset points')
            ax.set_xlabel(f'Reliability Score ({algorithm})', fontsize=13, fontweight='bold')
            ax.set_ylabel(f'R2', fontsize=13, fontweight='bold')
            ax.set_title(f'R2 vs Reliability ({algorithm})', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{dataset_name}_r2_vs_reliability_{algorithm.lower()}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_reliability_heatmap(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot heatmap of reliability scores (algorithms vs models).

        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        # Convert to DataFrame for easier plotting
        df_data = []
        for algorithm, model_scores in self.reliability_scores.items():
            for model, score in model_scores.items():
                df_data.append({'Algorithm': algorithm, 'Model': model, 'Score': score})

        if not df_data:
            print("No data to plot")
            return

        df = pd.DataFrame(df_data)
        pivot_df = df.pivot(index='Algorithm', columns='Model', values='Score')

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=0, vmax=100, cbar_kws={'label': 'Reliability Score (%)'},
                   linewidths=0.5, ax=ax)

        plt.title('Algorithm Reliability Scores by Model', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Generative Model', fontsize=12, fontweight='bold')
        plt.ylabel('Causal Discovery Algorithm', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_algorithm_comparison(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot bar chart comparing algorithms by average reliability.

        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        algo_ranking = self.rank_algorithms_by_reliability()

        if not algo_ranking:
            print("No algorithm rankings to plot")
            return

        algorithms = [name for name, _ in algo_ranking]
        scores = [score for _, score in algo_ranking]

        fig, ax = plt.subplots(figsize=figsize)

        # Color bars by score (green for high, red for low)
        colors = plt.cm.RdYlGn(np.array(scores) / 100)
        bars = ax.bar(algorithms, scores, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Causal Discovery Algorithm', fontsize=13, fontweight='bold')
        ax.set_ylabel('Average Reliability Score (%)', fontsize=13, fontweight='bold')
        ax.set_title('Algorithm Reliability Comparison (Higher is Better)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')

        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Algorithm comparison saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_model_ranking_by_algorithm(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)):
        """
        Plot model rankings for each algorithm.

        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        n_algorithms = len(self.ground_truths)
        if n_algorithms == 0:
            print("No algorithms to plot")
            return

        fig, axes = plt.subplots(1, n_algorithms, figsize=figsize, sharey=True)
        if n_algorithms == 1:
            axes = [axes]

        for idx, algorithm in enumerate(self.ground_truths.keys()):
            ranking = self.rank_models_by_algorithm(algorithm)

            if not ranking:
                continue

            models = [name for name, _ in ranking]
            scores = [score for _, score in ranking]

            ax = axes[idx]
            colors = plt.cm.RdYlGn(np.array(scores) / 100)
            bars = ax.barh(models, scores, color=colors, edgecolor='black', linewidth=1.5)

            # Add value labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{score:.1f}%',
                       ha='left', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_xlabel('Reliability Score (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{algorithm.upper()}', fontsize=13, fontweight='bold', pad=10)
            ax.set_xlim(0, 105)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        axes[0].set_ylabel('Generative Model', fontsize=12, fontweight='bold')
        fig.suptitle('Model Rankings by Algorithm (Higher is Better)',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Model rankings saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_cautabench_vs_reliability(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot scatter plot comparing CauTabBench scores vs average reliability scores.

        Args:
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.cautabench_scores:
            print("No CauTabBench scores to plot")
            return

        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        # Calculate average reliability for each model
        avg_reliability = {}
        for model_name in self.synthetic_structures.keys():
            scores = []
            for algorithm in self.reliability_scores.values():
                if model_name in algorithm:
                    scores.append(algorithm[model_name])
            if scores:
                avg_reliability[model_name] = np.mean(scores)

        # Prepare data
        models = []
        cautabench = []
        reliability = []

        for model in avg_reliability.keys():
            if model in self.cautabench_scores:
                models.append(model)
                cautabench.append(self.cautabench_scores[model])
                reliability.append(avg_reliability[model])

        if not models:
            print("No matching data to plot")
            return

        # Create scatter plot
        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(reliability, cautabench, s=200, alpha=0.6,
                           c=range(len(models)), cmap='tab10', edgecolors='black', linewidth=2)

        # Add labels for each point
        for i, model in enumerate(models):
            ax.annotate(model, (reliability[i], cautabench[i]),
                       fontsize=11, fontweight='bold',
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

        ax.set_xlabel('Average Algorithm Reliability Score (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('CauTabBench Quality Score', fontsize=13, fontweight='bold')
        ax.set_title('CauTabBench vs Algorithm Reliability', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add diagonal reference line
        min_val = min(min(reliability), min(cautabench) * 100)
        max_val = max(max(reliability), max(cautabench) * 100)
        ax.plot([min_val, max_val], [min_val/100, max_val/100], 'r--',
               alpha=0.3, linewidth=2, label='Perfect correlation')

        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_cautabench_vs_reliability_by_algorithm(self, algorithm: str, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot scatter plot comparing CauTabBench scores vs reliability scores for a specific algorithm.

        Args:
            algorithm: The specific causal discovery algorithm (e.g., 'pc', 'ges')
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.cautabench_scores:
            print("No CauTabBench scores to plot")
            return

        if not self.reliability_scores:
            self.compute_all_reliability_scores()

        if algorithm not in self.reliability_scores:
            print(f"No reliability scores for algorithm: {algorithm}")
            return

        # Get reliability scores for this specific algorithm
        algo_reliability = self.reliability_scores[algorithm]

        # Prepare data
        models = []
        cautabench = []
        reliability = []

        for model in algo_reliability.keys():
            if model in self.cautabench_scores:
                models.append(model)
                cautabench.append(self.cautabench_scores[model])
                reliability.append(algo_reliability[model])

        if not models:
            print(f"No matching data to plot for {algorithm}")
            return

        # Create scatter plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use different colors for different models
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        scatter = ax.scatter(reliability, cautabench, s=250, alpha=0.7,
                           c=[colors[i % len(colors)] for i in range(len(models))],
                           edgecolors='black', linewidth=2)

        # Add labels for each point
        for i, model in enumerate(models):
            ax.annotate(model.upper(), (reliability[i], cautabench[i]),
                       fontsize=12, fontweight='bold',
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

        ax.set_xlabel(f'Reliability Score according to {algorithm.upper()} (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('CauTabBench Quality Score', fontsize=13, fontweight='bold')
        ax.set_title(f'CauTabBench Quality Score vs Reliability ({algorithm.upper()})', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set axis limits with padding
        x_min, x_max = min(reliability) - 5, max(reliability) + 5
        y_min, y_max = min(cautabench) - 0.05, max(cautabench) + 0.05
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ {algorithm.upper()} comparison plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_all_visualizations(self, output_dir: str, dataset_name: str = "dataset"):
        """
        Generate and save all visualization plots.

        Args:
            output_dir: Directory to save all plots
            dataset_name: Name of the dataset for file naming
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating visualizations for {dataset_name}...")
        print(f"{'='*60}\n")

        # 1. Reliability heatmap
        heatmap_path = os.path.join(output_dir, f"{dataset_name}_reliability_heatmap.png")
        self.plot_reliability_heatmap(save_path=heatmap_path)

        # 2. Algorithm comparison
        algo_comp_path = os.path.join(output_dir, f"{dataset_name}_algorithm_comparison.png")
        self.plot_algorithm_comparison(save_path=algo_comp_path)

        # 3. Model rankings by algorithm
        model_rank_path = os.path.join(output_dir, f"{dataset_name}_model_rankings.png")
        self.plot_model_ranking_by_algorithm(save_path=model_rank_path)

        # 4. CauTabBench vs Reliability (if CauTabBench scores available)
        if self.cautabench_scores:
            # 4a. Average reliability (original)
            scatter_path = os.path.join(output_dir, f"{dataset_name}_cautabench_vs_reliability.png")
            self.plot_cautabench_vs_reliability(save_path=scatter_path)

            # 4b. Per-algorithm reliability plots (PC and GES)
            for algorithm in self.reliability_scores.keys():
                algo_scatter_path = os.path.join(output_dir, f"{dataset_name}_cautabench_vs_reliability_{algorithm.lower()}.png")
                self.plot_cautabench_vs_reliability_by_algorithm(algorithm=algorithm, save_path=algo_scatter_path)

        # 5. Regression metric vs reliability (R2 only)
        self.plot_regression_metric_vs_reliability("R2", output_dir, dataset_name)

        # 6. CI ROC AUC vs reliability
        self.plot_ci_auc_vs_reliability(output_dir, dataset_name)

        print(f"\n{'='*60}")
        print(f"✅ All visualizations saved to: {output_dir}")
        print(f"{'='*60}\n")

    def save(self, filepath: str):
        """Save ranking comparison results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        summary = self.summarize()

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Ranking comparison saved to: {filepath}")

    def load(self, filepath: str):
        """Load ranking comparison from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.reliability_scores = data.get('reliability_scores', {})
        print(f"✅ Ranking comparison loaded from: {filepath}")

    def plot_ci_auc_vs_reliability(self, output_dir: str, dataset_name: str):
        """
        Plot CI ROC AUC vs reliability score for each algorithm/model.
        Assumes CI AUC reports are saved as outputs/evaluations/{dataset}_{model}_ci_auc/ci_auc_report.json
        """
        if not self.reliability_scores:
            self.compute_all_reliability_scores()
        for algorithm in self.ground_truths.keys():
            models = []
            auc_vals = []
            reliability_vals = []
            for model in self.synthetic_structures.keys():
                auc_path = os.path.join('outputs', 'evaluations', f'{dataset_name}_{model}_ci_auc', 'ci_auc_report.json')
                if os.path.exists(auc_path) and model in self.reliability_scores[algorithm]:
                    with open(auc_path, 'r') as f:
                        auc = json.load(f).get('roc_auc', None)
                    if auc is not None:
                        models.append(model)
                        auc_vals.append(auc)
                        reliability_vals.append(self.reliability_scores[algorithm][model])
            if not models:
                continue
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(reliability_vals, auc_vals, s=200, edgecolors='black', alpha=0.7)
            for i, model in enumerate(models):
                ax.annotate(model, (reliability_vals[i], auc_vals[i]), fontsize=12, fontweight='bold', xytext=(10, 10), textcoords='offset points')
            ax.set_xlabel(f'Reliability Score ({algorithm})', fontsize=13, fontweight='bold')
            ax.set_ylabel('CI ROC AUC', fontsize=13, fontweight='bold')
            ax.set_title(f'CI ROC AUC vs Reliability ({algorithm})', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{dataset_name}_ci_auc_vs_reliability_{algorithm.lower()}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
