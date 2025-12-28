import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from src.structure_learning.ges import GESLearner

gmm_data = pd.read_csv('outputs/synthetic/asia_gmm_synthetic.csv')
print(f"Data shape: {gmm_data.shape}")
print(f"Columns: {list(gmm_data.columns)}")
print(f"\nVariance per column:")
print(gmm_data.var())
print(f"\nZero variance columns: {list(gmm_data.columns[gmm_data.var() < 1e-10])}")
print("\nRunning GES...")
learner = GESLearner()
try:
    graph = learner.fit(gmm_data)
    print(f"SUCCESS! Learned graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
except Exception as e:
    print(f"ERROR: {e}")
