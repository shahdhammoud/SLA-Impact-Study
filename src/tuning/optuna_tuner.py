import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional
import joblib
import os
import json
from sklearn.metrics import roc_auc_score

from src.evaluation.cautabbench_eval import CauTabBenchEvaluator
from src.evaluation.metrics import extract_independence_implications, ConditionalIndependenceTest
from src.evaluation.ci_auc_utils import compute_ci_auc
import networkx as nx


class OptunaTuner:

    def __init__(self, model_class, model_name: str, 
                 eval_metric: str = 'quality_score',
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 study_name: Optional[str] = None):

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

        def objective(trial):
            params = self._sample_params(trial, param_space)
            try:
                model = self.model_class(**params)
                if categorical_columns:
                    model.fit(train_data, categorical_columns=categorical_columns)
                else:
                    model.fit(train_data)
                synthetic_data = model.sample(len(val_data))
                synthetic_data = synthetic_data[val_data.columns]
                ci_auc, _, _ = compute_ci_auc(synthetic_data, self.feature_info, self.graph_path)
                print(f"[DEBUG] CI ROC AUC: {ci_auc}")
                return ci_auc
            except Exception as e:
                print(f"Trial failed with error: {e}")
                return 0.0

        self.study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials)
        }
    
    def _sample_params(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:

        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, list):
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'categorical')
                if param_type == 'categorical':
                    values = param_config.get('values', None)
                    if values is None and 'choices' in param_config:
                        values = param_config['choices']
                    if values is None:
                        raise ValueError(f"Categorical param '{param_name}' missing 'values' or 'choices': {param_config}")
                    params[param_name] = trial.suggest_categorical(param_name, values)
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
                else:
                    raise ValueError(f"Unknown param type for '{param_name}': {param_type}")
            else:
                raise ValueError(f"Invalid param config for '{param_name}': {param_config}")
        return params
    

    def save_study(self, filepath: str):

        if self.study is None:
            raise ValueError("No study to save. Run tune() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.study, filepath)
        
        params_file = filepath.replace('.pkl', '_best_params.json')
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
    
    def load_study(self, filepath: str):

        self.study = joblib.load(filepath)
        self.best_params = self.study.best_params
    
    def get_optimization_history(self) -> pd.DataFrame:

        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        df = self.study.trials_dataframe()
        return df
    
    def plot_optimization_history(self, filepath: str):

        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_param_importances(self, filepath: str):

        if self.study is None:
            raise ValueError("No study available. Run tune() first.")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()


def create_param_space_from_config(config: Dict[str, Any]) -> Dict[str, Any]:

    param_space = {}
    
    tuning_params = config.get('tuning_params', {})
    if 'tuning_params' in tuning_params and isinstance(tuning_params['tuning_params'], dict):
        tuning_params = tuning_params['tuning_params']
    for param_name, values in tuning_params.items():
        param_space[param_name] = values

    return param_space
