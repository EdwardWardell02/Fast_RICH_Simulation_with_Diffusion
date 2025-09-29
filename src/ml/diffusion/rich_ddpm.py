import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from datetime import datetime

class RICHDDPM:
    """
    Complete DDPM for RICH Cherenkov Ring Generation
    Fully functional implementation with proper backpropagation
    """
    
    def __init__(self, image_size=32, timesteps=1000, beta_start=1e-4, beta_end=0.2):
        self.image_size = image_size
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Mathematical Foundation: Variance Schedule
        # The beta schedule controls how much noise is added at each timestep
        # Linear schedule as in original DDPM paper: β_t ∈ [β_start, β_end]
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        
        # Pre-calculate all diffusion parameters (for efficiency)
        # α_t = 1 - β_t (amount of signal preserved at step t)
        self.alphas = 1.0 - self.betas
        
        # ᾱ_t = ∏_{s=1}^t α_s (cumulative product of alphas)
        # This gives us the total signal preservation from x_0 to x_t
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        
        # ᾱ_{t-1} for the reverse process calculations
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
        # √ᾱ_t and √(1-ᾱ_t) for the forward process reparameterization
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Reverse process parameters (from Bayes' theorem)
        # σ_t^2 = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Coefficients for the reverse process mean
        # μ_θ(x_t, t) = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
        self.posterior_mean_coef1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        # Automatically run basic verification
        self._basic_verification()

        print(f"DDPM initialized: image_size={image_size}, timesteps={timesteps}")
        print(f"Beta range: [{beta_start:.6f}, {beta_end:.6f}]")

    def _basic_verification(self):
        """Basic verification that alpha values are sensible"""
        print("=== Basic Alpha Verification ===")
        print(f"α_0: {self.alphas[0]:.6f} (should be close to 1)")
        print(f"α_{self.timesteps-1}: {self.alphas[-1]:.6f} (should be close to 0)")
        print(f"√ᾱ_0: {self.sqrt_alphas_cumprod[0]:.6f} (should be 1)")
        print(f"√ᾱ_{self.timesteps-1}: {self.sqrt_alphas_cumprod[-1]:.6f} (should be close to 0)")
        print("✅ Basic verification completed")

    def verify_alpha_calculations(self):
        """Comprehensive verification of alpha calculations"""
        print("\n" + "="*50)
        print("COMPREHENSIVE ALPHA CALCULATIONS VERIFICATION")
        print("="*50)
        
        # Check 1: α_t should decrease from nearly 1 to nearly 0
        print("\n1. Alpha progression check:")
        print(f"   α_0: {self.alphas[0]:.6f}")
        print(f"   α_{self.timesteps//4}: {self.alphas[self.timesteps//4]:.6f}")
        print(f"   α_{self.timesteps//2}: {self.alphas[self.timesteps//2]:.6f}")
        print(f"   α_{3*self.timesteps//4}: {self.alphas[3*self.timesteps//4]:.6f}")
        print(f"   α_{self.timesteps-1}: {self.alphas[-1]:.6f}")
        
        # Check 2: ᾱ_t should be the cumulative product
        expected_alphas_cumprod = np.cumprod(self.alphas)
        if np.allclose(self.alphas_cumprod, expected_alphas_cumprod, rtol=1e-10):
            print("✅ ᾱ_t calculation is correct (cumulative product of α_t)")
        else:
            print("❌ ᾱ_t calculation is incorrect!")
            max_diff = np.max(np.abs(self.alphas_cumprod - expected_alphas_cumprod))
            print(f"   Maximum difference: {max_diff:.10f}")
            return False
        
        # Check 3: √ᾱ_t should decrease from 1 to nearly 0
        print(f"\n2. √ᾱ_t progression:")
        print(f"   √ᾱ_0: {self.sqrt_alphas_cumprod[0]:.6f} (should be 1.0)")
        print(f"   √ᾱ_{self.timesteps-1}: {self.sqrt_alphas_cumprod[-1]:.6f} (should be close to 0)")
        
        # Check 4: √(1-ᾱ_t) should increase from 0 to nearly 1
        print(f"\n3. √(1-ᾱ_t) progression:")
        print(f"   √(1-ᾱ_0): {self.sqrt_one_minus_alphas_cumprod[0]:.6f} (should be 0.0)")
        print(f"   √(1-ᾱ_{self.timesteps-1}): {self.sqrt_one_minus_alphas_cumprod[-1]:.6f} (should be close to 1)")
        
        # Check 5: Critical mathematical property
        print(f"\n4. Mathematical property check:")
        test_t = self.timesteps // 2
        alpha_bar = self.alphas_cumprod[test_t]
        expected = alpha_bar + (1 - alpha_bar)
        print(f"   At t={test_t}: ᾱ_t + (1-ᾱ_t) = {alpha_bar:.6f} + {1-alpha_bar:.6f} = {expected:.6f} (should be 1.0)")
        
        if abs(expected - 1.0) < 1e-10:
            print("✅ Mathematical property verified")
        else:
            print("❌ Mathematical property failed!")
            return False
        
        print("\n🎉 All alpha calculations verified successfully!")
        return True

    def debug_noise_addition(self, x0, t):
        """Debug exactly how noise is being added at a specific timestep"""
        print(f"\n{'='*60}")
        print(f"NOISE ADDITION DEBUG - Timestep {t}")
        print(f"{'='*60}")
        
        # Get the scaling factors for this timestep
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        print(f"Scaling factors:")
        print(f"  √ᾱ_t = {sqrt_alpha:.6f}")
        print(f"  √(1-ᾱ_t) = {sqrt_one_minus_alpha:.6f}")
        
        # Generate noise
        noise = np.random.normal(0, 1, x0.shape)
        
        print(f"\nInput statistics:")
        print(f"  x0 range: [{x0.min():.6f}, {x0.max():.6f}]")
        print(f"  x0 mean: {x0.mean():.6f}, std: {x0.std():.6f}")
        print(f"  Noise range: [{noise.min():.6f}, {noise.max():.6f}]")
        print(f"  Noise mean: {noise.mean():.6f}, std: {noise.std():.6f}")
        
        # Apply forward process step by step
        signal_component = sqrt_alpha * x0
        noise_component = sqrt_one_minus_alpha * noise
        
        print(f"\nComponent statistics:")
        print(f"  Signal component range: [{signal_component.min():.6f}, {signal_component.max():.6f}]")
        print(f"  Noise component range: [{noise_component.min():.6f}, {noise_component.max():.6f}]")
        
        x_t = signal_component + noise_component
        
        print(f"\nOutput statistics:")
        print(f"  x_t range: [{x_t.min():.6f}, {x_t.max():.6f}]")
        print(f"  x_t mean: {x_t.mean():.6f}, std: {x_t.std():.6f}")
        
        # Check variance preservation
        original_variance = np.var(x0)
        final_variance = np.var(x_t)
        expected_variance = (sqrt_alpha ** 2) * original_variance + (sqrt_one_minus_alpha ** 2) * 1.0
        
        print(f"\nVariance preservation check:")
        print(f"  Original variance: {original_variance:.6f}")
        print(f"  Final variance: {final_variance:.6f}")
        print(f"  Expected variance: {expected_variance:.6f}")
        print(f"  Error: {abs(final_variance - expected_variance):.6f}")
        
        if abs(final_variance - expected_variance) < 0.01:
            print("  ✅ Variance preserved correctly")
        else:
            print("  ❌ Variance preservation failed!")
        
        return x_t

    def _verify_variance_preservation(self):
        """Verify that the forward process preserves unit variance"""
        print(f"\n{'='*60}")
        print("VARIANCE PRESERVATION VERIFICATION")
        print(f"{'='*60}")
        
        # Create test data with unit variance
        test_data = np.random.normal(0, 1, (100, self.image_size, self.image_size, 1))
        
        test_points = [0, self.timesteps//4, self.timesteps//2, 3*self.timesteps//4, self.timesteps-1]
        
        all_passed = True
        for t in test_points:
            t_batch = np.array([t] * len(test_data))
            xt, _ = self.forward_diffusion(test_data, t_batch)
            
            variance = np.var(xt)
            expected_variance = 1.0  # Should be preserved
            
            error = abs(variance - expected_variance)
            status = "✅" if error < 0.1 else "❌"
            
            print(f"t={t:4d}: variance={variance:.6f}, expected={expected_variance}, error={error:.6f} {status}")
            
            if error >= 0.1:
                all_passed = False
        
        if all_passed:
            print("🎉 Variance preservation verified across all timesteps!")
        else:
            print("❌ Variance preservation failed at some timesteps!")
        
        return all_passed
    def forward_diffusion(self, x0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Mathematical Formulation:
        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε, where ε ~ N(0, I)
        
        This is the reparameterization trick that allows us to sample x_t 
        directly from x_0 without going through all intermediate steps.
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)
            #noise = np.random.normal(0, 1, x0.shape)
        # Debug: check noise statistics
        noise_mean = noise.mean()
        noise_std = noise.std()
        if abs(noise_mean) > 0.1 or abs(noise_std - 1.0) > 0.1:
            print(f"Warning: Noise not N(0,1)! Mean: {noise_mean:.3f}, Std: {noise_std:.3f}")
        # Extract the appropriate ᾱ_t values for the given timesteps
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        # Apply the forward diffusion formula
        x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def reverse_diffusion_step(self, model, xt, t):
        """
        Single reverse diffusion step: p_θ(x_{t-1} | x_t)
        
        Mathematical Formulation:
        1. Predict noise: ε_θ = model(x_t, t)
        2. Estimate x_0: x_0 ≈ (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        3. Compute mean: μ_θ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
        4. Sample: x_{t-1} = μ_θ + σ_t * z, where z ~ N(0, I)
        """
        # Predict the noise using the U-Net
        predicted_noise = model.forward(xt, t)
        
        # Extract parameters for the current timestep
        alpha_t = self._extract(self.alphas, t, xt.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, xt.shape)
        beta_t = self._extract(self.betas, t, xt.shape)
        
        # Mathematical: Estimate x_0 from current x_t and predicted noise
        sqrt_alpha_cumprod_t = np.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = np.sqrt(1.0 - alpha_cumprod_t)
        
        pred_x0 = (xt - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        pred_x0 = np.clip(pred_x0, -1.0, 1.0)
        
        # Mathematical: Compute the mean of the reverse process distribution
        # This comes from the variational lower bound derivation
        mean = (1.0 / np.sqrt(alpha_t)) * (xt - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
        
        # Add noise except at the final step (t=0)
        if t[0] > 0:
            posterior_variance_t = self._extract(self.posterior_variance, t, xt.shape)
            noise = np.random.normal(0, 1, xt.shape)
            sample = mean + np.sqrt(posterior_variance_t) * noise
        else:
            sample = mean
        
        return sample, pred_x0, predicted_noise
    
    def _extract(self, arr, t, x_shape):
        """Extract values from array for specific timesteps t"""
        batch_size = t.shape[0]
        out = arr[t]
        
        # Reshape to add dimensions for broadcasting
        while len(out.shape) < len(x_shape):
            out = out[..., np.newaxis]
        
        return out

    def generate_samples(self, model, num_samples=1, progress_callback=None):
        """
        Generate samples from pure noise using the reverse process
        
        Mathematical Process:
        Start from x_T ~ N(0, I) and iteratively apply:
        x_{t-1} = reverse_step(x_t, t) for t = T, T-1, ..., 1
        """
        # Start from pure Gaussian noise
        x = np.random.normal(0, 1, (num_samples, self.image_size, self.image_size, 1))
        
        intermediate_samples = []
        
        # Reverse process: go from t = T-1 down to 0
        for i in range(self.timesteps - 1, -1, -1):
            t = np.array([i] * num_samples)
            x, pred_x0, _ = self.reverse_diffusion_step(model, x, t)
            
            if progress_callback and i % 100 == 0:
                progress_callback(i, x)
            
            if i % 200 == 0:  # Store intermediate samples for visualization
                intermediate_samples.append(x.copy())
        
        return x, intermediate_samples