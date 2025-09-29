import torch
import numpy as np

class RICHDDPM:
    """PyTorch-compatible DDPM implementation"""
    
    def __init__(self, image_size=32, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='auto'):
        self.image_size = image_size
        self.timesteps = timesteps
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Create variance schedule on the correct device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), 
                                            self.alphas_cumprod[:-1]])
        
        # Precompute values for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for reverse process
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        print(f"DDPM initialized on device: {self.device}")
    
    def forward_diffusion(self, x0, t, noise=None):
        """Forward diffusion process with PyTorch tensors"""
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise
    
    def reverse_diffusion_step(self, model, xt, t):
        """Reverse diffusion step with PyTorch tensors"""
        # Predict noise
        predicted_noise = model(xt, t)
        
        # Extract parameters
        alpha_t = self._extract(self.alphas, t, xt.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, xt.shape)
        beta_t = self._extract(self.betas, t, xt.shape)
        
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        
        # Predict x0
        pred_x0 = (xt - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Calculate mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
        
        if t[0] > 0:
            posterior_variance_t = self._extract(self.posterior_variance, t, xt.shape)
            noise = torch.randn_like(xt, device=self.device)
            sample = mean + torch.sqrt(posterior_variance_t) * noise
        else:
            sample = mean
        
        return sample, pred_x0, predicted_noise
    
    def _extract(self, arr, t, shape):
        """Extract values from array for specific timesteps"""
        batch_size = t.shape[0]
        out = arr[t]
        # Add dimensions to match shape
        while len(out.shape) < len(shape):
            out = out.unsqueeze(-1)
        return out
    
    def generate_samples(self, model, num_samples=1, device='auto'):
        """Generate samples using PyTorch"""
        if device == 'auto':
            device = self.device
        
        # Start from noise
        x = torch.randn((num_samples, 1, self.image_size, self.image_size), device=device)
        
        intermediate_samples = []
        
        for i in range(self.timesteps - 1, -1, -1):
            t = torch.tensor([i] * num_samples, device=device)
            x, pred_x0, _ = self.reverse_diffusion_step(model, x, t)
            
            if i % 200 == 0:
                intermediate_samples.append(x.clone())
        
        return x, intermediate_samples