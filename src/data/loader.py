"""
Data loader for benchmark datasets with ground truth causal structures.

This module handles loading CSV data files and TXT structure files.
"""

import os
import pandas as pd
import networkx as nx
from typing import Tuple, Dict, Optional, List


class DataLoader:
    """Load benchmark datasets with ground truth structures."""
    
    def __init__(self, base_path: str = "benchmarks_with_ground_truth"):
        """
        Initialize DataLoader.
        
        Args:
            base_path: Base directory containing csv/ and txt/ subdirectories
        """
        self.base_path = base_path
        self.csv_path = os.path.join(base_path, "csv")
        self.txt_path = os.path.join(base_path, "txt")
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, nx.DiGraph]:
        """
        Load both data and ground truth structure for a dataset.
        
        Args:
            dataset_name: Name of the dataset (without extension)
            
        Returns:
            Tuple of (dataframe, directed graph)
        """
        data = self.load_data(dataset_name)
        structure = self.load_structure(dataset_name)
        return data, structure
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load CSV data file.
        
        Args:
            dataset_name: Name of the dataset (without .csv extension)
            
        Returns:
            DataFrame with the dataset
        """
        csv_file = os.path.join(self.csv_path, f"{dataset_name}.csv")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Load CSV with header
        df = pd.read_csv(csv_file)
        
        # Check if first column might be an index
        first_col = df.columns[0]
        if first_col.lower() in ['index', 'id', 'unnamed: 0'] or df[first_col].equals(pd.Series(range(len(df)))):
            df = df.drop(columns=[first_col])
        
        return df
    
    def load_structure(self, dataset_name: str) -> nx.DiGraph:
        """
        Load ground truth causal structure from TXT file.
        
        The TXT file should contain edges as space-separated node names:
        A C  (means edge A -> C)
        B C  (means edge B -> C)
        
        Args:
            dataset_name: Name of the dataset (without .txt extension)
            
        Returns:
            NetworkX directed graph representing the causal structure
        """
        txt_file = os.path.join(self.txt_path, f"{dataset_name}.txt")
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Structure file not found: {txt_file}")
        
        G = nx.DiGraph()
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) == 2:
                    source, target = parts
                    G.add_edge(source, target)
                else:
                    raise ValueError(f"Invalid edge format: {line}. Expected 'source target'")
        
        return G
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names (without extensions)
        """
        datasets = []
        
        if os.path.exists(self.csv_path):
            csv_files = [f[:-4] for f in os.listdir(self.csv_path) 
                        if f.endswith('.csv') and not f.startswith('.')]
            
            # Only include datasets that have both CSV and TXT files
            for dataset in csv_files:
                txt_file = os.path.join(self.txt_path, f"{dataset}.txt")
                if os.path.exists(txt_file):
                    datasets.append(dataset)
        
        return sorted(datasets)
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get metadata about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        data = self.load_data(dataset_name)
        structure = self.load_structure(dataset_name)
        
        return {
            'name': dataset_name,
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'n_edges': structure.number_of_edges(),
            'n_nodes': structure.number_of_nodes(),
            'nodes': list(structure.nodes()),
        }
