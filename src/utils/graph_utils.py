"""
Graph and structure utilities for causal discovery and evaluation.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt


def adjacency_matrix_to_graph(adj_matrix: np.ndarray, 
                              node_names: List[str] = None) -> nx.DiGraph:
    """
    Convert adjacency matrix to NetworkX directed graph.
    
    Args:
        adj_matrix: Adjacency matrix (i,j)=1 means edge i->j
        node_names: Optional list of node names
        
    Returns:
        NetworkX DiGraph
    """
    n_nodes = adj_matrix.shape[0]
    
    if node_names is None:
        node_names = [f"X{i}" for i in range(n_nodes)]
    
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] != 0:
                G.add_edge(node_names[i], node_names[j])
    
    return G


def graph_to_adjacency_matrix(G: nx.DiGraph, 
                              node_names: List[str] = None) -> np.ndarray:
    """
    Convert NetworkX graph to adjacency matrix.
    
    Args:
        G: NetworkX directed graph
        node_names: Optional ordered list of node names
        
    Returns:
        Adjacency matrix
    """
    if node_names is None:
        node_names = sorted(list(G.nodes()))
    
    n_nodes = len(node_names)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    node_to_idx = {node: idx for idx, node in enumerate(node_names)}
    
    for source, target in G.edges():
        i = node_to_idx[source]
        j = node_to_idx[target]
        adj_matrix[i, j] = 1
    
    return adj_matrix


def compute_shd(G_true: nx.DiGraph, G_learned: nx.DiGraph, return_percentage: bool = True) -> float:
    """
    Compute Structural Hamming Distance between two graphs.
    
    SHD counts edge additions, deletions, and reversals needed to
    transform learned graph to true graph.
    
    Args:
        G_true: True causal graph
        G_learned: Learned causal graph
        return_percentage: If True, return SHD as percentage of possible edges (default: True)
        
    Returns:
        Structural Hamming Distance (as percentage if return_percentage=True, otherwise absolute value)
    """
    true_edges = set(G_true.edges())
    learned_edges = set(G_learned.edges())
    
    # Missing edges (in true but not in learned)
    missing = len(true_edges - learned_edges)
    
    # Extra edges (in learned but not in true)
    extra = len(learned_edges - true_edges)
    
    # Check for reversed edges
    reversed_edges = 0
    for u, v in learned_edges - true_edges:
        if (v, u) in true_edges:
            reversed_edges += 1
    
    # SHD = additions + deletions + reversals
    # Reversed edges are counted in both missing and extra, so subtract once
    shd = missing + extra - reversed_edges
    
    if return_percentage:
        # Calculate as percentage of possible edges
        n_nodes = len(G_true.nodes())
        max_possible_edges = n_nodes * (n_nodes - 1)  # Maximum possible directed edges
        if max_possible_edges > 0:
            shd_percentage = (shd / max_possible_edges) * 100.0
            return shd_percentage
        else:
            return 0.0
    
    return float(shd)


def compute_graph_metrics(G_true: nx.DiGraph, G_learned: nx.DiGraph) -> Dict[str, float]:
    """
    Compute various metrics comparing true and learned graphs.
    
    Args:
        G_true: True causal graph
        G_learned: Learned causal graph
        
    Returns:
        Dictionary of metrics (SHD as percentage, precision, recall, F1)
    """
    true_edges = set(G_true.edges())
    learned_edges = set(G_learned.edges())
    
    # True positives: edges in both
    tp = len(true_edges & learned_edges)
    
    # False positives: edges in learned but not in true
    fp = len(learned_edges - true_edges)
    
    # False negatives: edges in true but not in learned
    fn = len(true_edges - learned_edges)
    
    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # SHD as percentage
    shd_percentage = compute_shd(G_true, G_learned, return_percentage=True)
    
    # Also compute absolute SHD for reference
    shd_absolute = compute_shd(G_true, G_learned, return_percentage=False)
    
    return {
        'shd': shd_percentage,  # Now returns percentage
        'shd_absolute': shd_absolute,  # Keep absolute value for reference
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_true_edges': len(true_edges),
        'n_learned_edges': len(learned_edges),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def save_graph_visualization(G: nx.DiGraph, filepath: str, title: str = "Causal Graph"):
    """
    Save graph visualization to file.
    
    Args:
        G: NetworkX directed graph
        filepath: Path to save image
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use hierarchical layout if possible
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
            node_size=1500, font_size=10, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray', 
            arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def get_markov_blanket(G: nx.DiGraph, node: str) -> Set[str]:
    """
    Get Markov blanket of a node in the graph.
    
    Markov blanket includes:
    - Parents of the node
    - Children of the node
    - Parents of children (co-parents)
    
    Args:
        G: NetworkX directed graph
        node: Node name
        
    Returns:
        Set of nodes in the Markov blanket
    """
    if node not in G.nodes():
        raise ValueError(f"Node {node} not in graph")
    
    mb = set()
    
    # Add parents
    mb.update(G.predecessors(node))
    
    # Add children and their parents
    children = list(G.successors(node))
    mb.update(children)
    
    for child in children:
        mb.update(G.predecessors(child))
    
    # Remove the node itself
    mb.discard(node)
    
    return mb


def get_conditional_independencies(G: nx.DiGraph) -> List[Tuple[str, str, Set[str]]]:
    """
    Extract conditional independence relationships from graph.
    
    Returns list of (X, Y, Z) tuples representing X _|_ Y | Z
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        List of conditional independence tuples
    """
    nodes = list(G.nodes())
    independencies = []
    
    for i, x in enumerate(nodes):
        for y in nodes[i+1:]:
            # Check if X and Y are d-separated given various conditioning sets
            # This is a simplified version - full d-separation is more complex
            
            mb_x = get_markov_blanket(G, x)
            mb_y = get_markov_blanket(G, y)
            
            # If Y not in X's Markov blanket, they may be independent given MB(X)
            if y not in mb_x and x not in mb_y:
                conditioning_set = mb_x
                independencies.append((x, y, conditioning_set))
    
    return independencies


def is_dag(G: nx.DiGraph) -> bool:
    """
    Check if graph is a Directed Acyclic Graph.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        True if DAG, False otherwise
    """
    return nx.is_directed_acyclic_graph(G)
