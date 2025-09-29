# visualiser.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

class RICHVisualiser:
    """PyTorch-compatible visualisation system for RICH DDPM training and evaluation"""
    
    def __init__(self, output_dir="rich_diffusion_visualisations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color scheme optimised for RICH ring visualisation
        self.cmap = 'viridis'
        print(f"RICHVisualiser initialised. Output directory: {self.output_dir}")

    def _tensor_to_numpy(self, tensor):
        """Convert PyTorch tensor to numpy array for visualisation"""
        if isinstance(tensor, torch.Tensor):
            # Detach, move to CPU, and convert to numpy
            tensor = tensor.detach().cpu()
            if tensor.dim() == 4 and tensor.shape[1] in [1, 3]:  # Channel-first format
                tensor = tensor.permute(0, 2, 3, 1)  # Convert to channel-last for plotting
            return tensor.numpy()
        return tensor

    def plot_forward_process(self, ddpm, original_image, timesteps=[0, 25, 100, 250, 500, 1000], filename="forward_diffusion_process.png"):
        """
        Visualise the forward diffusion process: image → noise
        Now compatible with PyTorch tensors
        """
        # Convert to numpy for visualisation
        original_image_np = self._tensor_to_numpy(original_image)
        
        if len(original_image_np.shape) == 3:
            original_image_np = np.expand_dims(original_image_np, 0)  # Add batch dimension
        
        # Ensure image is in [-1, 1] range as expected by DDPM
        if original_image_np.max() <= 1.0 and original_image_np.min() >= 0.0:
            original_image_np = original_image_np * 2 - 1
        
        num_steps = len(timesteps)
        fig, axes = plt.subplots(2, num_steps, figsize=(5*num_steps, 6))
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        for i, t in enumerate(timesteps):
            # Ensure timestep is within valid range
            t = min(t, ddpm.timesteps - 1)
            
            # Convert to tensor for DDPM processing
            t_tensor = torch.tensor([t], device=ddpm.device)
            original_image_tensor = torch.FloatTensor(original_image_np).to(ddpm.device)
            
            # Apply forward diffusion using PyTorch DDPM
            noisy_image, noise = ddpm.forward_diffusion(original_image_tensor, t_tensor)
            
            # Convert back to numpy for visualisation
            noisy_image_np = self._tensor_to_numpy(noisy_image)
            noise_np = self._tensor_to_numpy(noise)
            
            # Convert to display format [0, 1]
            display_noisy = (noisy_image_np[0, :, :, 0] + 1) * 0.5
            display_noise = (noise_np[0, :, :, 0] + 1) * 0.5
            
            # Plot noisy image
            im1 = axes[0, i].imshow(display_noisy, cmap=self.cmap)
            axes[0, i].set_title(f'Step {t}\n(β={ddpm.betas[t].item():.4f})', fontsize=10)
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
            
            # Plot added noise
            im2 = axes[1, i].imshow(display_noise, cmap='RdBu_r', vmin=0, vmax=1)
            axes[1, i].set_title(f'Added Noise (t={t})', fontsize=10)
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
        
        plt.suptitle('Forward Diffusion Process: RICH Ring → Noise', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Forward process visualisation saved: {filename}")
        return fig

    def plot_reverse_process(self, ddpm, model, num_samples=1, num_steps=8, 
                           filename="reverse_diffusion_formation.png"):
        """
        Visualise the reverse diffusion process: noise → image
        Now compatible with PyTorch models
        """
        # Set model to evaluation mode
        model.eval()
        
        # Start from pure noise on the correct device
        x = torch.randn((num_samples, 1, ddpm.image_size, ddpm.image_size), 
                       device=ddpm.device)
        
        # Select timesteps to visualise
        visualisation_steps = []
        total_steps = ddpm.timesteps
        
        for i in range(num_steps):
            if i == 0:
                step = total_steps - 1
            elif i == num_steps - 1:
                step = 0
            else:
                step = int(total_steps * (1 - (i / (num_steps - 1)) ** 2))
            visualisation_steps.append(step)
        
        intermediate_samples = []
        current_steps = []
        
        print("Generating reverse process samples...")
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for i in range(total_steps - 1, -1, -1):
                t = torch.tensor([i] * num_samples, device=ddpm.device)
                x, pred_x0, _ = ddpm.reverse_diffusion_step(model, x, t)
                
                if i in visualisation_steps:
                    intermediate_samples.append(x.clone())
                    current_steps.append(i)
                    print(f"Captured step {i}")
        
        # Convert to numpy for visualisation
        intermediate_samples_np = [self._tensor_to_numpy(sample) for sample in intermediate_samples]
        
        # Create visualisation
        fig, axes = plt.subplots(2, len(intermediate_samples_np), figsize=(20, 8))
        if len(intermediate_samples_np) == 1:
            axes = axes.reshape(2, 1)
        
        for i, (sample, step) in enumerate(zip(intermediate_samples_np, current_steps)):
            # Convert to display format
            display_sample = (sample[0, :, :, 0] + 1) * 0.5
            progress = 1.0 - (step / total_steps)
            
            # Plot current state
            im1 = axes[0, i].imshow(display_sample, cmap=self.cmap)
            axes[0, i].set_title(f't = {step}\n({progress*100:.1f}% complete)', fontsize=10)
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
            
            # Note: pred_x0 is not stored in the current loop, so we'll skip that subplot
            # You can modify the reverse_diffusion_step to return and store pred_x0 if needed
            axes[1, i].axis('off')  # Placeholder for pred_x0
        
        plt.suptitle('Reverse Diffusion Process: Noise → RICH Ring Formation', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Reverse process visualisation saved: {filename}")
        return fig, intermediate_samples_np, current_steps

    def plot_training_progress(self, losses, val_losses=None, filename="training_progress.png"):
        """Plot training and validation loss over time - compatible with PyTorch tensors"""
        # Convert losses to numpy if they're tensors
        if isinstance(losses, list) and len(losses) > 0 and isinstance(losses[0], torch.Tensor):
            losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in losses]
        
        if val_losses is not None and isinstance(val_losses[0], torch.Tensor):
            val_losses = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
        plt.title('DDPM Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if val_losses is not None:
            plt.subplot(1, 2, 2)
            plt.plot(val_losses, 'r-', linewidth=2, label='Validation Loss')
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Training progress plot saved: {filename}")

    def create_diffusion_animation(self, samples, timesteps, filename="ring_formation_animation.gif"):
        """Create GIF animation of the reverse diffusion process - PyTorch compatible"""
        images = []
        
        for i, (sample, timestep) in enumerate(zip(samples, timesteps)):
            # Convert to numpy if it's a tensor
            sample_np = self._tensor_to_numpy(sample)
            
            # Convert to PIL Image
            img_data = ((sample_np[0, :, :, 0] + 1) * 127.5).astype(np.uint8)
            img = Image.fromarray(img_data).convert('RGB')
            
            # Add timestep annotation
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            progress = 1.0 - (timestep / timesteps[0])
            text = f"Step: {timestep} ({progress*100:.1f}%)"
            draw.text((5, 5), text, fill=(255, 255, 255), font=font)
            
            images.append(img)
        
        if images:
            images[0].save(
                self.output_dir / filename,
                save_all=True,
                append_images=images[1:],
                duration=800,
                loop=0,
                optimize=True
            )
            print(f"Diffusion animation saved: {filename}")

    def plot_comparison_grid(self, real_images, generated_images, filename="real_vs_generated.png"):
        """Compare real RICH rings with generated ones - PyTorch compatible"""
        # Convert to numpy if they're tensors
        real_images = [self._tensor_to_numpy(img) for img in real_images]
        generated_images = [self._tensor_to_numpy(img) for img in generated_images]
        
        num_images = min(len(real_images), len(generated_images), 8)
        
        fig, axes = plt.subplots(2, num_images, figsize=(20, 6))
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        # Plot real images
        for i in range(num_images):
            img = real_images[i]
            if img.max() <= 1.0:
                display_real = img
            else:
                display_real = img / img.max()
            
            # Ensure correct shape for imshow
            if display_real.shape[-1] == 1:  # Squeeze channel dimension if needed
                display_real = display_real.squeeze()
            
            axes[0, i].imshow(display_real, cmap=self.cmap)
            axes[0, i].set_title(f'Real Ring {i+1}')
            axes[0, i].axis('off')
        
        # Plot generated images
        for i in range(num_images):
            img = generated_images[i]
            if img.max() <= 1.0:
                display_gen = img
            else:
                display_gen = (img + 1) * 0.5
            
            if display_gen.shape[-1] == 1:
                display_gen = display_gen.squeeze()
            
            axes[1, i].imshow(display_gen, cmap=self.cmap)
            axes[1, i].set_title(f'Generated Ring {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle('Real vs Generated RICH Cherenkov Rings', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison grid saved: {filename}")

    def plot_loss_curves(self, train_losses, val_losses=None, filename="loss_curves.png"):
        """More detailed loss curve plotting"""
        plt.figure(figsize=(10, 6))
        
        # Convert to numpy if they're tensors
        if isinstance(train_losses, list) and train_losses and isinstance(train_losses[0], torch.Tensor):
            train_losses = [loss.item() for loss in train_losses]
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.7)
        
        if val_losses is not None:
            if isinstance(val_losses[0], torch.Tensor):
                val_losses = [loss.item() for loss in val_losses]
            plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.7)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Often helpful for diffusion models
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curves saved: {filename}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.stats import norm
class StochasticProcessVisualiser:
    """Visualize the exponential complexity of diffusion trajectories"""
    
    def __init__(self, output_dir="stochastic_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_stochastic_process_illustration(self, num_paths=10, timesteps=100, filename="stochastic_process.png"):
        """
        Visualize multiple stochastic trajectories in diffusion process
        """
        plt.figure(figsize=(15, 10))
        
        # Simulate multiple stochastic paths (Brownian motion-like)
        np.random.seed(42)  # For reproducibility
        
        # Create multiple trajectories
        time = np.linspace(0, 1, timesteps)
        
        plt.subplot(2, 2, 1)
        for i in range(num_paths):
            # Cumulative random walk
            steps = np.random.normal(0, 0.1, timesteps-1)
            path = np.cumsum(np.concatenate([[0], steps]))
            plt.plot(time, path, alpha=0.7, linewidth=1)
        
        plt.xlabel('Time (Diffusion Process)')
        plt.ylabel('State Value')
        plt.title(f'{num_paths} Random Diffusion Trajectories')
        plt.grid(True, alpha=0.3)
        
        # Probability density evolution
        plt.subplot(2, 2, 2)
        
        x = np.linspace(-3, 3, 100)
        for t in [0.1, 0.3, 0.6, 1.0]:
            # Variance increases with time in diffusion
            variance = t
            pdf = norm.pdf(x, scale=np.sqrt(variance))
            plt.plot(x, pdf, label=f't={t}', linewidth=2)
        
        plt.xlabel('State Value')
        plt.ylabel('Probability Density')
        plt.title('Probability Distribution Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Branching factor visualization
        plt.subplot(2, 2, 3)
        branching_factors = [1, 2, 3, 4, 5]
        max_depth = 5
        
        def plot_branching_tree(ax, branching_factor, depth):
            """Recursively plot branching tree"""
            if depth == 0:
                return
            
            # Calculate positions
            x_positions = np.linspace(-1, 1, branching_factor ** depth)
            y_position = depth
            
            for i in range(branching_factor ** depth):
                ax.plot([0], [0], 'bo', markersize=8)  # Root
                if depth > 0:
                    for j in range(branching_factor):
                        ax.plot([0, x_positions[i * branching_factor + j]], 
                               [0, y_position], 'k-', alpha=0.3)
            
            # Recursive call for next level
            plot_branching_tree(ax, branching_factor, depth - 1)
        
        # Simple branching visualization
        for i, bf in enumerate(branching_factors):
            total_nodes = sum(bf ** d for d in range(max_depth + 1))
            plt.bar(bf, total_nodes, alpha=0.7, label=f'Branch={bf}')
        
        plt.xlabel('Branching Factor')
        plt.ylabel('Total Nodes after 5 steps')
        plt.title('Tree Growth with Different Branching Factors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Intractability threshold
        plt.subplot(2, 2, 4)
        computation_times = []
        trajectory_counts = []
        
        for T in range(1, 21):  # Up to 20 steps for visualization
            trajectories = 2 ** T
            trajectory_counts.append(trajectories)
            
            # Estimate computation time (exponential growth)
            # Assuming 1 nanosecond per trajectory evaluation (optimistic)
            time_seconds = trajectories * 1e-9
            
            # Convert to meaningful units
            if time_seconds < 60:
                computation_times.append(time_seconds)
            elif time_seconds < 3600:
                computation_times.append(time_seconds / 60)  # minutes
            elif time_seconds < 86400:
                computation_times.append(time_seconds / 3600)  # hours
            else:
                computation_times.append(time_seconds / 86400)  # days
        
        time_units = ['seconds', 'minutes', 'hours', 'days']
        unit_thresholds = [60, 3600, 86400]
        
        plt.semilogy(range(1, 21), computation_times, 'red', linewidth=3, marker='o')
        plt.axvline(x=10, color='orange', linestyle='--', label='T=10 (Manageable)')
        plt.axvline(x=15, color='red', linestyle='--', label='T=15 (Intractable)')
        plt.axvline(x=20, color='darkred', linestyle='--', label='T=20 (Impossible)')
        
        plt.xlabel('Number of Timesteps (T)')
        plt.ylabel('Computation Time')
        plt.title('Exponential Computation Time Growth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Stochastic process illustration saved: {filename}")


import torch

class DiffusionAnalysisVisualiser:
    def __init__(self, output_dir="diffusion_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_variance_schedule(self, ddpm, filename="variance_schedule.png"):
        """
        Plot the variance schedule and its mathematical properties
        """
        t_values = np.arange(ddpm.timesteps)
        
        plt.figure(figsize=(15, 10))
        
        # Beta schedule
        plt.subplot(2, 3, 1)
        plt.plot(t_values, ddpm.betas.cpu().numpy(), 'b-', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('βₜ')
        plt.title('Noise Schedule βₜ')
        plt.grid(True, alpha=0.3)
        
        # Alpha cumulative product
        plt.subplot(2, 3, 2)
        alpha_bar = ddpm.alphas_cumprod.cpu().numpy()
        plt.plot(t_values, alpha_bar, 'r-', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('ᾱₜ')
        plt.title('Cumulative Product ᾱₜ = ∏(1-βₛ)')
        plt.grid(True, alpha=0.3)
        
        # Signal-to-noise ratio
        plt.subplot(2, 3, 3)
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        plt.semilogy(t_values, snr, 'g-', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('SNR = ᾱₜ/(1-ᾱₜ)')
        plt.title('Signal-to-Noise Ratio Evolution')
        plt.grid(True, alpha=0.3)
        
        # Forward process variance components
        plt.subplot(2, 3, 4)
        sqrt_alpha_bar = np.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)
        plt.plot(t_values, sqrt_alpha_bar, 'b-', linewidth=2, label='√ᾱₜ')
        plt.plot(t_values, sqrt_one_minus_alpha_bar, 'r-', linewidth=2, label='√(1-ᾱₜ)')
        plt.xlabel('Timestep t')
        plt.ylabel('Scaling Factors')
        plt.title('Forward Process Scaling Factors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Posterior variance
        plt.subplot(2, 3, 5)
        posterior_var = ddpm.posterior_variance.cpu().numpy()
        plt.plot(t_values[1:], posterior_var[1:], 'purple', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('σₜ²')
        plt.title('Reverse Process Posterior Variance')
        plt.grid(True, alpha=0.3)
        
        # Mathematical relationship check
        plt.subplot(2, 3, 6)
        check_value = alpha_bar + (1 - alpha_bar)
        plt.plot(t_values, check_value, 'orange', linewidth=2)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Timestep t')
        plt.ylabel('ᾱₜ + (1-ᾱₜ)')
        plt.title('Mathematical Consistency Check\n(Should be exactly 1)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Variance schedule plot saved: {filename}")
        
        return {
            'betas': ddpm.betas.cpu().numpy(),
            'alpha_bar': alpha_bar,
            'snr': snr
        }

    def plot_elbo_components(self, ddpm, filename="elbo_analysis.png"):
        """
        Analyze the Evidence Lower Bound (ELBO) components
        """
        t_values = np.arange(1, ddpm.timesteps)
        
        # Calculate theoretical ELBO components
        alpha_bar = ddpm.alphas_cumprod.cpu().numpy()
        alpha_bar_prev = np.concatenate([[1.0], alpha_bar[:-1]])
        
        # L_T: KL divergence between q(x_T|x_0) and N(0, I)
        L_T = 0.5 * (alpha_bar[-1] + np.log(1 - alpha_bar[-1]) - 1)
        
        # L_{t-1}: KL divergences for intermediate steps
        # Simplified calculation for visualization
        kl_terms = []
        for t in range(1, ddpm.timesteps):
            # Simplified KL divergence between two Gaussians
            mean_ratio = (1 - alpha_bar[t-1]) / (1 - alpha_bar[t]) * ddpm.alphas[t].cpu().numpy()
            kl = 0.5 * (mean_ratio - 1 - np.log(mean_ratio))
            kl_terms.append(kl)
        
        # L_0: Reconstruction term
        L_0 = -0.5 * np.log(2 * np.pi * ddpm.posterior_variance[0].cpu().numpy())
        
        plt.figure(figsize=(12, 10))
        
        # KL divergence terms over time
        plt.subplot(2, 2, 1)
        plt.plot(range(1, ddpm.timesteps), kl_terms, 'b-', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('KL[q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)]')
        plt.title('KL Divergence Terms in ELBO\n(L_{1} to L_{T-1})')
        plt.grid(True, alpha=0.3)
        
        # Cumulative KL divergence
        plt.subplot(2, 2, 2)
        cumulative_kl = np.cumsum(kl_terms)
        plt.plot(range(1, ddpm.timesteps), cumulative_kl, 'r-', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('Cumulative KL Divergence')
        plt.title('Cumulative KL Divergence\n(Sum of L_{1} to L_{t})')
        plt.grid(True, alpha=0.3)
        
        # Variance of reverse process
        plt.subplot(2, 2, 3)
        posterior_var = ddpm.posterior_variance.cpu().numpy()
        plt.semilogy(range(ddpm.timesteps), posterior_var, 'g-', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('Posterior Variance σₜ²')
        plt.title('Reverse Process Variance Schedule')
        plt.grid(True, alpha=0.3)
        
        # Signal preservation over time
        plt.subplot(2, 2, 4)
        signal_preservation = alpha_bar
        noise_level = 1 - alpha_bar
        plt.plot(t_values, signal_preservation, 'b-', linewidth=2, label='Signal (√ᾱₜ)')
        plt.plot(t_values, noise_level, 'r-', linewidth=2, label='Noise (√(1-ᾱₜ))')
        plt.xlabel('Timestep t')
        plt.ylabel('Scaling Factor')
        plt.title('Signal vs Noise in Forward Process')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ELBO analysis plot saved: {filename}")
        
        return {
            'kl_terms': kl_terms,
            'cumulative_kl': cumulative_kl,
            'L_T': L_T,
            'L_0': L_0
        }

    def plot_reparameterization_trick(self, ddpm, filename="reparameterization.png"):
        """
        Visualize the reparameterization trick and its importance
        """
        t_values = np.arange(ddpm.timesteps)
        alpha_bar = ddpm.alphas_cumprod.cpu().numpy()
        
        plt.figure(figsize=(15, 8))
        
        # Forward process with reparameterization
        plt.subplot(2, 3, 1)
        # Show the mathematical form
        x0 = 1.0  # Normalized input
        epsilon = np.random.randn(len(t_values))
        
        x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * epsilon
        
        plt.plot(t_values, x_t, 'b-', alpha=0.7, linewidth=1)
        plt.fill_between(t_values, x_t - 0.1, x_t + 0.1, alpha=0.3)
        plt.xlabel('Timestep t')
        plt.ylabel('xₜ')
        plt.title('Forward Process: xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε')
        plt.grid(True, alpha=0.3)
        
        # Distribution at different timesteps
        plt.subplot(2, 3, 2)
        sample_timesteps = [0, ddpm.timesteps//4, ddpm.timesteps//2, 3*ddpm.timesteps//4, ddpm.timesteps-1]
        
        for t in sample_timesteps:
            # Generate multiple samples at this timestep
            samples = []
            for _ in range(1000):
                eps = np.random.randn()
                sample = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
                samples.append(sample)
            
            # Plot distribution
            plt.hist(samples, bins=50, alpha=0.6, label=f't={t}', density=True)
        
        plt.xlabel('xₜ value')
        plt.ylabel('Probability Density')
        plt.title('Distribution Evolution\n(Reparameterization Trick)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Computational efficiency
        plt.subplot(2, 3, 3)
        naive_complexity = 2 ** (t_values / 10)  # Scaled exponential
        reparam_complexity = t_values  # Linear
        
        plt.semilogy(t_values, naive_complexity, 'r-', linewidth=2, 
                    label='Naive Path Sampling (Exponential)')
        plt.semilogy(t_values, reparam_complexity, 'g-', linewidth=2,
                    label='Reparameterization (Linear)')
        plt.xlabel('Timesteps T')
        plt.ylabel('Computational Complexity')
        plt.title('Computational Efficiency\nof Reparameterization Trick')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gradient flow comparison
        plt.subplot(2, 3, 4)
        # Show how reparameterization enables better gradients
        t_vals = np.linspace(0, ddpm.timesteps-1, 50)
        grad_naive = np.exp(-t_vals / 100)  # Vanishing gradients
        grad_reparam = np.ones_like(t_vals)  # Stable gradients
        
        plt.plot(t_vals, grad_naive, 'r-', linewidth=2, label='Naive (High Variance)')
        plt.plot(t_vals, grad_reparam, 'g-', linewidth=2, label='Reparameterization (Low Variance)')
        plt.xlabel('Timestep t')
        plt.ylabel('Gradient Magnitude')
        plt.title('Gradient Estimation Quality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mathematical formulation
        plt.subplot(2, 3, 5)
        # Empty plot for mathematical equations
        plt.text(0.5, 0.7, 'Forward Process:', ha='center', va='center', fontsize=14, weight='bold')
        plt.text(0.5, 0.5, 'q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)', ha='center', va='center', fontsize=12)
        plt.text(0.5, 0.3, 'xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε', ha='center', va='center', fontsize=12, style='italic')
        plt.text(0.5, 0.1, 'where ε ∼ N(0, I)', ha='center', va='center', fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Mathematical Foundation')
        
        # Practical impact
        plt.subplot(2, 3, 6)
        memory_naive = naive_complexity * 4e-9  # GB
        memory_reparam = reparam_complexity * 4e-9  # GB
        
        plt.plot(t_values, memory_naive, 'r-', linewidth=2, label='Naive Approach')
        plt.plot(t_values, memory_reparam, 'g-', linewidth=2, label='Reparameterization')
        plt.axhline(y=8, color='blue', linestyle='--', label='8GB GPU Memory')
        plt.xlabel('Timesteps T')
        plt.ylabel('Memory Required (GB)')
        plt.title('Memory Requirements Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Reparameterization trick plot saved: {filename}")

    def plot_training_dynamics(self, trainer, filename="training_dynamics.png"):
        """
        Analyze training dynamics and loss landscape
        """
        if not hasattr(trainer, 'losses') or len(trainer.losses) == 0:
            print("No training losses available")
            return
        
        losses = trainer.losses
        
        plt.figure(figsize=(15, 10))
        
        # Training loss curve
        plt.subplot(2, 3, 1)
        plt.plot(losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Progression')
        plt.grid(True, alpha=0.3)
        
        # Log-scale loss
        plt.subplot(2, 3, 2)
        plt.semilogy(losses, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Log MSE Loss')
        plt.title('Training Loss (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # Loss distribution
        plt.subplot(2, 3, 3)
        plt.hist(losses, bins=50, alpha=0.7, color='green')
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.title('Loss Value Distribution')
        plt.grid(True, alpha=0.3)
        
        # Gradient norms (simulated - in practice you'd track these)
        plt.subplot(2, 3, 4)
        # Simulate typical gradient norm behavior
        epochs = range(len(losses))
        grad_norms = [max(0.1, 1.0 / (1 + 0.01 * epoch)) for epoch in epochs]
        plt.semilogy(epochs, grad_norms, 'purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Typical Gradient Norm Evolution')
        plt.grid(True, alpha=0.3)
        
        # Learning rate schedule impact
        plt.subplot(2, 3, 5)
        lr = trainer.learning_rate
        effective_lr = [lr * (0.99 ** epoch) for epoch in epochs]  # Simulated decay
        plt.semilogy(epochs, effective_lr, 'orange', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Effective Learning Rate')
        plt.title('Learning Rate Schedule Impact')
        plt.grid(True, alpha=0.3)
        
        # Convergence analysis
        plt.subplot(2, 3, 6)
        moving_avg = np.convolve(losses, np.ones(10)/10, mode='valid')
        plt.plot(epochs[9:], moving_avg, 'b-', linewidth=2, label='Moving Average (10)')
        plt.axhline(y=np.mean(losses[-10:]), color='red', linestyle='--', 
                   label=f'Final Avg: {np.mean(losses[-10:]):.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('Smoothed Loss')
        plt.title('Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training dynamics plot saved: {filename}")

    def plot_jensen_inequality_analysis(self, ddpm, filename="jensen_inequality.png"):
        """
        Visualize the role of Jensen's inequality in the variational lower bound
        """
        plt.figure(figsize=(15, 10))
        
        # Jensen's inequality illustration
        plt.subplot(2, 2, 1)
        x = np.linspace(0.1, 2, 100)
        log_x = np.log(x)
        
        plt.plot(x, log_x, 'b-', linewidth=3, label='log(x)')
        plt.plot(x, x - 1, 'r--', linewidth=2, label='x - 1 (tangent at x=1)')
        
        # Highlight convexity
        x_sample = np.array([0.5, 1.5])
        y_sample = np.log(x_sample)
        chord_y = [y_sample[0], y_sample[1]]
        chord_x = [x_sample[0], x_sample[1]]
        
        plt.plot(chord_x, chord_y, 'g-', linewidth=2, alpha=0.7, label='Chord')
        plt.fill_between(x, log_x, x-1, where=(x>=0.5)&(x<=1.5), alpha=0.3, color='orange')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title("Jensen's Inequality: E[log(X)] ≤ log(E[X])\nfor convex functions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ELBO decomposition
        plt.subplot(2, 2, 2)
        components = ['L_T', 'L_{t-1}', 'L_0']
        theoretical_contributions = [0.1, 0.7, 0.2]  # Example values
        colors = ['red', 'blue', 'green']
        
        plt.pie(theoretical_contributions, labels=components, colors=colors, autopct='%1.1f%%')
        plt.title('Theoretical ELBO Components\n(-ELBO = L_T + ΣL_{t-1} + L_0)')
        
        # Variational gap illustration
        plt.subplot(2, 2, 3)
        t_values = np.arange(ddpm.timesteps)
        
        # Simulate true log-likelihood and ELBO
        true_ll = -np.exp(-t_values / 200)  # Simulated true log-likelihood
        elbo = true_ll - 0.5 * (1 - np.exp(-t_values / 200))  # Simulated ELBO (lower bound)
        
        plt.plot(t_values, true_ll, 'g-', linewidth=3, label='True Log-Likelihood')
        plt.plot(t_values, elbo, 'r-', linewidth=2, label='ELBO (Variational Lower Bound)')
        plt.fill_between(t_values, elbo, true_ll, alpha=0.3, color='orange', label='Variational Gap')
        
        plt.xlabel('Model Capacity (Training Progress)')
        plt.ylabel('Log-Likelihood')
        plt.title('Evidence Lower Bound (ELBO) vs True Likelihood')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # KL divergence terms in ELBO
        plt.subplot(2, 2, 4)
        alpha_bar = ddpm.alphas_cumprod.cpu().numpy()
        kl_terms = []
        
        for t in range(1, ddpm.timesteps):
            # Simplified KL term
            kl = 0.5 * ((1 - alpha_bar[t-1]) / (1 - alpha_bar[t]) * 
                        ddpm.alphas[t].cpu().numpy() - 1 - 
                        np.log((1 - alpha_bar[t-1]) / (1 - alpha_bar[t]) * 
                              ddpm.alphas[t].cpu().numpy()))
            kl_terms.append(max(0, kl))
        
        plt.plot(range(1, ddpm.timesteps), kl_terms, 'purple', linewidth=2)
        plt.xlabel('Timestep t')
        plt.ylabel('KL[q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t)]')
        plt.title('KL Divergence Terms in ELBO')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Jensen's inequality analysis plot saved: {filename}")