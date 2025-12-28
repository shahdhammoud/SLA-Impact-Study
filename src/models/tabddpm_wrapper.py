import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from .base import BaseGenerativeModel


class MLPDiffusion(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4, 
                 dropout: float = 0.0, timesteps: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.timesteps = timesteps
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
        self.hidden_layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        h = self.input_layer(x)
        h = h + t_emb
        h = self.hidden_layers(h)
        out = self.output_layer(h)
        
        return out


class TabDDPMWrapper(BaseGenerativeModel):

    def __init__(self, **kwargs):
        super().__init__(model_name="TabDDPM", **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() and kwargs.get('use_gpu', True) else 'cpu'
        
        default_params = {
            'num_timesteps': 1000,
            'gaussian_loss_type': 'mse',
            'scheduler': 'cosine',
            'model_type': 'mlp',
            'num_layers': 4,
            'hidden_dim': 256,
            'dropout': 0.0,
            'lr': 0.002,
            'weight_decay': 1e-4,
            'batch_size': 1024,
            'epochs': 1000,
            'device': self.device
        }
        
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        self.diffusion_model = None
        self.data_mean = None
        self.data_std = None
        self.training_losses = []
        
    def _get_beta_schedule(self, schedule_type: str = 'cosine') -> torch.Tensor:
        timesteps = self.params['num_timesteps']
        
        if schedule_type == 'linear':
            beta = torch.linspace(1e-4, 0.02, timesteps)
        elif schedule_type == 'cosine':
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1)
            alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta = torch.clip(betas, 0.0001, 0.9999)
        else:
            beta = torch.linspace(1e-4, 0.02, timesteps)
        
        return beta
    
    def _setup_diffusion(self, input_dim: int):
        self.betas = self._get_beta_schedule(self.params['scheduler']).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.diffusion_model = MLPDiffusion(
            input_dim=input_dim,
            hidden_dim=self.params['hidden_dim'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            timesteps=self.params['num_timesteps']
        ).to(self.device)
    
    def fit(self, data: pd.DataFrame, **kwargs):
        self.feature_names = list(data.columns)
        
        data_np = data.values.astype(np.float32)
        self.data_mean = np.mean(data_np, axis=0)
        self.data_std = np.std(data_np, axis=0) + 1e-6
        data_normalized = (data_np - self.data_mean) / self.data_std
        
        input_dim = data_normalized.shape[1]
        self._setup_diffusion(input_dim)
        
        tensor_data = torch.FloatTensor(data_normalized)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)
        
        optimizer = optim.Adam(
            self.diffusion_model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )
        
        self.training_losses = []
        self.diffusion_model.train()
        
        for epoch in range(self.params['epochs']):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                x_0 = batch[0].to(self.device)
                batch_size = x_0.shape[0]
                
                t = torch.randint(0, self.params['num_timesteps'], (batch_size,), device=self.device)
                
                noise = torch.randn_like(x_0)
                
                sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
                sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
                x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
                
                t_input = t.float().view(-1, 1) / self.params['num_timesteps']
                predicted_noise = self.diffusion_model(x_t, t_input)
                
                if self.params['gaussian_loss_type'] == 'mse':
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                else:
                    loss = nn.functional.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {avg_loss:.6f}")
        
        self.model = self.diffusion_model
        self.is_fitted = True
        
        self.metadata = {
            'n_samples': len(data),
            'n_features': len(data.columns),
            'features': list(data.columns),
            'data_mean': self.data_mean.tolist(),
            'data_std': self.data_std.tolist(),
            'training_losses': self.training_losses
        }
    
    def sample(self, n_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples using TabDDPM.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            DataFrame with synthetic samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before sampling")
        
        self.diffusion_model.eval()
        
        with torch.no_grad():
            x = torch.randn(n_samples, len(self.feature_names), device=self.device)
            
            for t in reversed(range(self.params['num_timesteps'])):
                t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                t_input = t_batch.float().view(-1, 1) / self.params['num_timesteps']
                
                predicted_noise = self.diffusion_model(x, t_input)
                
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * noise
        
        x_np = x.cpu().numpy()
        x_denormalized = x_np * self.data_std + self.data_mean
        
        synthetic_data = pd.DataFrame(x_denormalized, columns=self.feature_names)
        
        return synthetic_data
    
    def get_training_losses(self) -> List[float]:
        return self.training_losses

    def save(self, path: str):
        import pickle

        save_dict = {
            'model_state_dict': self.diffusion_model.state_dict() if self.diffusion_model else None,
            'feature_names': self.feature_names,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'params': self.params,
            'metadata': self.metadata,
            'training_losses': self.training_losses,
            'betas': self.betas,
            'alphas': self.alphas,
            'alphas_cumprod': self.alphas_cumprod,
            'sqrt_alphas_cumprod': self.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, path: str):
        import pickle

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.feature_names = save_dict['feature_names']
        self.data_mean = save_dict['data_mean']
        self.data_std = save_dict['data_std']
        self.params = save_dict['params']
        self.metadata = save_dict['metadata']
        self.training_losses = save_dict['training_losses']
        self.betas = save_dict['betas']
        self.alphas = save_dict['alphas']
        self.alphas_cumprod = save_dict['alphas_cumprod']
        self.sqrt_alphas_cumprod = save_dict['sqrt_alphas_cumprod']
        self.sqrt_one_minus_alphas_cumprod = save_dict['sqrt_one_minus_alphas_cumprod']

        input_dim = len(self.feature_names)
        self.diffusion_model = MLPDiffusion(
            input_dim=input_dim,
            hidden_dim=self.params['hidden_dim'],
            num_layers=self.params['num_layers'],
            dropout=self.params['dropout'],
            timesteps=self.params['num_timesteps']
        ).to(self.device)

        if save_dict['model_state_dict']:
            self.diffusion_model.load_state_dict(save_dict['model_state_dict'])

        self.model = self.diffusion_model
        self.is_fitted = True
