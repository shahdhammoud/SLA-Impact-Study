"""
Visualization utilities for training losses and rankings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os


def plot_training_loss(losses: List[float], model_name: str, 
                       save_path: Optional[str] = None, 
                       title: Optional[str] = None) -> None:
    """
    Plot training loss over epochs.
    
    Args:
        losses: List of loss values
        model_name: Name of the model
        save_path: Path to save the plot (if None, plot is displayed)
        title: Custom title for the plot
    """
    if not losses:
        print(f"No training losses to plot for {model_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    if len(losses) > 1:
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch', fontsize=12)
    else:
        # For models with single score (GMM, Bayesian Network)
        plt.bar([model_name], losses, color='steelblue')
        plt.ylabel('Score', fontsize=12)
    
    if title is None:
        title = f'{model_name} - Training Loss'
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    if len(losses) > 1:
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_training_losses(losses_dict: Dict[str, List[float]], 
                                  save_path: Optional[str] = None,
                                  title: str = "Training Losses Comparison") -> None:
    """
    Plot training losses for multiple models.
    
    Args:
        losses_dict: Dictionary mapping model names to loss lists
        save_path: Path to save the plot
        title: Title for the plot
    """
    if not losses_dict:
        print("No training losses to plot")
        return
    
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']
    
    for i, (model_name, losses) in enumerate(losses_dict.items()):
        if len(losses) > 1:
            epochs = range(1, len(losses) + 1)
            color = colors[i % len(colors)]
            plt.plot(epochs, losses, color=color, linewidth=2, 
                    label=model_name, marker='o', markersize=3, alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined training loss plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_rankings(rankings_dict: Dict[str, List[Tuple[str, float]]], 
                       save_path: Optional[str] = None,
                       title: str = "Model Rankings Across Structures") -> None:
    """
    Plot model rankings across different structure learning algorithms.
    
    Args:
        rankings_dict: Dictionary mapping structure names to rankings
                      Each ranking is a list of (model_name, score) tuples
        save_path: Path to save the plot
        title: Title for the plot
    """
    if not rankings_dict:
        print("No rankings to plot")
        return
    
    # Extract all unique models
    all_models = set()
    for rankings in rankings_dict.values():
        all_models.update([model for model, _ in rankings])
    all_models = sorted(list(all_models))
    
    # Create a matrix of rankings (position for each model under each structure)
    structures = list(rankings_dict.keys())
    n_structures = len(structures)
    n_models = len(all_models)
    
    # Position matrix (lower position = better rank)
    position_matrix = np.zeros((n_models, n_structures))
    score_matrix = np.zeros((n_models, n_structures))
    
    for j, structure in enumerate(structures):
        rankings = rankings_dict[structure]
        model_to_rank = {model: i for i, (model, _) in enumerate(rankings)}
        model_to_score = {model: score for model, score in rankings}
        
        for i, model in enumerate(all_models):
            position_matrix[i, j] = model_to_rank.get(model, n_models)
            score_matrix[i, j] = model_to_score.get(model, 0.0)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Ranking positions (heatmap)
    im1 = ax1.imshow(position_matrix, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(range(n_structures))
    ax1.set_yticks(range(n_models))
    ax1.set_xticklabels(structures, rotation=45, ha='right')
    ax1.set_yticklabels(all_models)
    ax1.set_title('Model Rankings (Position)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Structure Learning Algorithm', fontsize=10)
    ax1.set_ylabel('Model', fontsize=10)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_structures):
            text = ax1.text(j, i, f'{int(position_matrix[i, j]) + 1}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Rank Position')
    
    # Plot 2: Scores (bar chart)
    x = np.arange(n_structures)
    width = 0.8 / n_models
    
    colors_bar = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for i, model in enumerate(all_models):
        offset = (i - n_models/2) * width
        ax2.bar(x + offset, score_matrix[i, :], width, label=model, 
               color=colors_bar[i], alpha=0.8)
    
    ax2.set_xlabel('Structure Learning Algorithm', fontsize=10)
    ax2.set_ylabel('Quality Score', fontsize=10)
    ax2.set_title('Model Quality Scores', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(structures, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ranking visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ranking_comparison(baseline_ranking: List[Tuple[str, float]], 
                           comparison_ranking: List[Tuple[str, float]],
                           baseline_name: str = "Ground Truth",
                           comparison_name: str = "Learned Structure",
                           save_path: Optional[str] = None) -> None:
    """
    Plot comparison between two rankings.
    
    Args:
        baseline_ranking: Baseline ranking as list of (model, score) tuples
        comparison_ranking: Comparison ranking as list of (model, score) tuples
        baseline_name: Name of baseline structure
        comparison_name: Name of comparison structure
        save_path: Path to save the plot
    """
    models = [model for model, _ in baseline_ranking]
    baseline_scores = [score for _, score in baseline_ranking]
    
    # Reorder comparison scores to match baseline order
    comparison_dict = {model: score for model, score in comparison_ranking}
    comparison_scores = [comparison_dict.get(model, 0.0) for model in models]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Side-by-side bar chart
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_scores, width, label=baseline_name, 
           color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, comparison_scores, width, label=comparison_name,
           color='coral', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Quality Score', fontsize=11)
    ax1.set_title('Quality Score Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Rank position comparison
    baseline_ranks = list(range(1, len(models) + 1))
    comparison_models = [model for model, _ in comparison_ranking]
    comparison_ranks = [comparison_models.index(model) + 1 if model in comparison_models else len(models) 
                       for model in models]
    
    for i, model in enumerate(models):
        ax2.plot([0, 1], [baseline_ranks[i], comparison_ranks[i]], 
                'o-', linewidth=2, markersize=8, label=model, alpha=0.7)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([baseline_name, comparison_name])
    ax2.set_ylabel('Rank Position', fontsize=11)
    ax2.set_title('Rank Position Changes', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ranking comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_consensus_ranking(rankings_dict: Dict[str, List[Tuple[str, float]]],
                          consensus_ranking: List[Tuple[str, float]],
                          save_path: Optional[str] = None) -> None:
    """
    Plot consensus ranking vs individual structure rankings.
    
    Args:
        rankings_dict: Dictionary mapping structure names to rankings
        consensus_ranking: Consensus ranking across all structures
        save_path: Path to save the plot
    """
    models = [model for model, _ in consensus_ranking]
    consensus_scores = [score for _, score in consensus_ranking]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot consensus as bars
    x = np.arange(len(models))
    ax.bar(x, consensus_scores, color='gold', alpha=0.7, label='Consensus', 
          edgecolor='black', linewidth=1.5)
    
    # Plot individual structure rankings as scatter points
    colors = plt.cm.Set3(np.linspace(0, 1, len(rankings_dict)))
    
    for i, (structure, rankings) in enumerate(rankings_dict.items()):
        score_dict = {model: score for model, score in rankings}
        scores = [score_dict.get(model, 0.0) for model in models]
        ax.scatter(x, scores, s=100, alpha=0.6, label=structure, 
                  color=colors[i], edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Quality Score', fontsize=12)
    ax.set_title('Consensus Ranking vs Individual Structure Rankings', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Consensus ranking plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
