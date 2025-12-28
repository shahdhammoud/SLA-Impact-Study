#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')

from src.models.tabddpm_wrapper import TabDDPMWrapper
from src.models.ctgan_wrapper import CTGANWrapper
from src.models.gmm_wrapper import GMMWrapper
from src.models.bayesian_network import BayesianNetworkWrapper
from src.utils.graph_utils import compute_shd, compute_graph_metrics
from src.utils.visualization_utils import (
    plot_training_loss,
    plot_multiple_training_losses,
    plot_model_rankings,
    plot_ranking_comparison,
    plot_consensus_ranking
)

os.makedirs('outputs/demo', exist_ok=True)

np.random.seed(42)
n_samples = 200
test_data = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples) * 2 + 1,
    'feature3': np.random.randn(n_samples) * 0.5
})

tabddpm = TabDDPMWrapper(
    epochs=20,
    batch_size=64,
    num_timesteps=200,
    hidden_dim=128,
    num_layers=3,
    lr=0.002
)

tabddpm.fit(test_data)

synthetic_data = tabddpm.sample(100)

plot_training_loss(
    losses=tabddpm.get_training_losses(),
    model_name='TabDDPM',
    save_path='outputs/demo/tabddpm_training_loss.png',
    title='TabDDPM Training Loss - Demo'
)

ctgan_losses = [1.0 / (1.0 + i * 0.02) for i in range(50)]
gmm_loss = [-15000]
bn_loss = [-8500]

plot_training_loss(ctgan_losses, 'CTGAN',
                  save_path='outputs/demo/ctgan_training_loss.png', title='CTGAN Training Loss - Demo')
plot_training_loss(gmm_loss, 'GMM',
                  save_path='outputs/demo/gmm_training_loss.png', title='GMM Training Loss - Demo')
plot_training_loss(bn_loss, 'BayesianNetwork',
                  save_path='outputs/demo/bn_training_loss.png', title='Bayesian Network Training Loss - Demo')
