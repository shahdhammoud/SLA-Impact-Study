"""
Optuna-based hyperparameter tuning for generative models.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional
import joblib
import os
import json

from src.evaluation.cautabbench_eval import CauTabBenchEvaluator
import networkx as nx


class OptunaTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(self, model_class, model_name: str, 
                 eval_metric: str = 'quality_score',
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 study_name: Optional[str] = None):
        """
        Initialize Optuna tuner.
        
        Args:
            model_class: Generative model class
            model_name: Name of the model
            eval_metric: Metric to optimize
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            study_name: Name for the Optuna study
        """
        self.model_class = model_class
        self.model_name = model_name
        self.eval_metric = eval_metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"{model_name}_tuning"
        self.study = None
        self.best_params = None
    
    def tune(self, train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            causal_graph: nx.DiGraph,
            param_space: Dict[str, Any],
            categorical_columns: Optional[list] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            train_data: Training data
            val_data: Validation data
            causal_graph: Causal structure for evaluation
            param_space: Parameter search space
            categorical_columns: List of categorical columns
            
        Returns:
            Best parameters and results
        """
        # Create objective function
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params(trial, param_space)
            
            # Train model
            try:
                model = self.model_class(**params)
                
                if categorical_columns:
                    model.fit(train_data, categorical_columns=categorical_columns)
                else:
                    model.fit(train_data)
                
                # Generate synthetic data
                synthetic_data = model.sample(len(val_data))
                
                # Evaluate using CauTabBench methodology
                evaluator = CauTabBenchEvaluator()
                results = evaluator.evaluate(val_data, synthetic_data, causal_graph)
                
                # Return metric to optimize
                return results[self.eval_metric]
                
            except Exception as e:
                print(f"Trial failed with error: {e}")
                return -1.0  # Return low score for failed trials
        
        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials)
        }
    
    def _sample_params(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample hyperparameters from search space.
        
        Args:
            trial: Optuna trial
            param_space: Parameter search space definition
            
        Returns:
            Sampled parameters
        """
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'categorical')
                
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['values']
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config['low'], 
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
        
        return params
    
    def save_study(self, filepath: str):
        """
        Save Optuna study to file.
        
        Args:
            filepath: Path to save study
        """
        if self.study is None:
            raise ValueError("No study to save. Run tune() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save study
        joblib.dump(self.study, filepath)
        
        # Also save best params as JSON for easy access
        params_file = filepath.replace('.pkl', '_best_params.json')
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
    
    def load_study(self, filepath: str):
        """
        Load Optuna study from file.
        
        Args:
            filepath: Path to load study from
        """
        self.study = joblib.load(filepath)
        self.best_params = self.study.best_params
    
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        df = self.study.trials_dataframe()
        return df
    
    def plot_optimization_history(self, filepath: str):
        """
        Plot and save optimization history.
        
        Args:
            filepath: Path to save plot
        """
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_param_importances(self, filepath: str):
        """
        Plot and save parameter importances.
        
        Args:
            filepath: Path to save plot
        """
        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()


def create_param_space_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert configuration to Optuna parameter space.
    
    Args:
        config: Configuration dictionary with tuning_params
        
    Returns:
        Parameter space for Optuna
    """
    param_space = {}
    
    tuning_params = config.get('tuning_params', {})
    
    for param_name, values in tuning_params.items():
        if isinstance(values, list):
            param_space[param_name] = values
        else:
            param_space[param_name] = values
    
    return param_space
