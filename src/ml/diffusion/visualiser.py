import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

class RICHVisualiser:
    """Complete visualisation system for RICH DDPM training and evaluation"""
    
    def __init__(self, output_dir="rich_diffusion_visualisations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color scheme optimised for RICH ring visualisation
        self.cmap = 'viridis'
        print(f"RICHVisualiser initialised. Output directory: {self.output_dir}")

    def plot_forward_process(self, ddpm, original_image, num_steps=6, filename="forward_diffusion_process.png"):
        """
        Visualise the forward diffusion process: image → noise :cite[2]:cite[5]
        
        Parameters:
        - ddpm: RICHDDPM instance
        - original_image: Single RICH image from your dataset (32x32x1)
        - num_steps: Number of diffusion steps to visualise
        """
        if len(original_image.shape) == 3:
            original_image = np.expand_dims(original_image, 0)  # Add batch dimension
        
        # Ensure image is in [-1, 1] range as expected by DDPM
        if original_image.max() <= 1.0 and original_image.min() >= 0.0:
            original_image = original_image * 2 - 1
        
        fig, axes = plt.subplots(2, num_steps, figsize=(20, 6))
        if num_steps == 1:
            axes = axes.reshape(2, 1)
        
        # Select timesteps to visualise
        timesteps = np.linspace(0, ddpm.timesteps - 1, num_steps, dtype=int)
        
        for i, t in enumerate(timesteps):
            # Convert to batch format
            t_batch = np.array([t])
            
            # Apply forward diffusion
            noisy_image, noise = ddpm.forward_diffusion(original_image, t_batch)
            
            # Convert back to display format [0, 1]
            display_noisy = (noisy_image[0, :, :, 0] + 1) * 0.5
            display_noise = (noise[0, :, :, 0] + 1) * 0.5
            
            # Plot noisy image
            im1 = axes[0, i].imshow(display_noisy, cmap=self.cmap)
            axes[0, i].set_title(f'Step {t}\n(β={ddpm.betas[t]:.4f})', fontsize=10)
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
        Visualise the reverse diffusion process: noise → image :cite[5]:cite[9]
        
        Parameters:
        - ddpm: RICHDDPM instance
        - model: Trained UNet model
        - num_samples: Number of samples to generate
        - num_steps: Number of reverse steps to visualise
        """
        # Start from pure noise
        x = np.random.normal(0, 1, (num_samples, ddpm.image_size, ddpm.image_size, 1))
        
        # Select timesteps to visualise (more frequent at the end where changes are dramatic)
        visualisation_steps = []
        total_steps = ddpm.timesteps
        
        # More samples toward the end where image forms quickly
        for i in range(num_steps):
            if i == 0:
                step = total_steps - 1
            elif i == num_steps - 1:
                step = 0
            else:
                # Logarithmic spacing to capture more detail at the end
                step = int(total_steps * (1 - (i / (num_steps - 1)) ** 2))
            visualisation_steps.append(step)
        
        intermediate_samples = []
        current_steps = []
        
        print("Generating reverse process samples...")
        for i in range(total_steps - 1, -1, -1):
            t = np.array([i] * num_samples)
            x, pred_x0, _ = ddpm.reverse_diffusion_step(model, x, t)
            
            if i in visualisation_steps:
                intermediate_samples.append(x.copy())
                current_steps.append(i)
                print(f"Captured step {i}")
        
        # Create visualisation
        fig, axes = plt.subplots(2, len(intermediate_samples), figsize=(20, 8))
        if len(intermediate_samples) == 1:
            axes = axes.reshape(2, 1)
        
        for i, (sample, step) in enumerate(zip(intermediate_samples, current_steps)):
            # Convert to display format
            display_sample = (sample[0, :, :, 0] + 1) * 0.5
            progress = 1.0 - (step / total_steps)
            
            # Plot current state
            im1 = axes[0, i].imshow(display_sample, cmap=self.cmap)
            axes[0, i].set_title(f't = {step}\n({progress*100:.1f}% complete)', fontsize=10)
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)
            
            # Plot predicted clean image
            pred_display = (pred_x0[0, :, :, 0] + 1) * 0.5
            im2 = axes[1, i].imshow(pred_display, cmap=self.cmap)
            axes[1, i].set_title(f'Predicted x₀ (t={step})', fontsize=10)
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046)
        
        plt.suptitle('Reverse Diffusion Process: Noise → RICH Ring Formation', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Reverse process visualisation saved: {filename}")
        return fig, intermediate_samples, current_steps

    def plot_training_progress(self, losses, val_losses=None, filename="training_progress.png"):
        """Plot training and validation loss over time"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
        plt.title('DDPM Training Loss')
        plt.xlabel('Training Step')
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
        """Create GIF animation of the reverse diffusion process"""
        images = []
        
        for i, (sample, timestep) in enumerate(zip(samples, timesteps)):
            # Convert to PIL Image
            img_data = ((sample[0, :, :, 0] + 1) * 127.5).astype(np.uint8)
            img = Image.fromarray(img_data).convert('RGB')
            
            # Add timestep annotation
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Use default font (you might need to adjust this)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            progress = 1.0 - (timestep / timesteps[0])
            text = f"Step: {timestep} ({progress*100:.1f}%)"
            draw.text((5, 5), text, fill=(255, 255, 255), font=font)
            
            images.append(img)
        
        if images:
            # Save as GIF
            images[0].save(
                self.output_dir / filename,
                save_all=True,
                append_images=images[1:],
                duration=800,  # milliseconds per frame
                loop=0,  # infinite loop
                optimize=True
            )
            print(f"Diffusion animation saved: {filename}")

    def plot_comparison_grid(self, real_images, generated_images, filename="real_vs_generated.png"):
        """Compare real RICH rings with generated ones"""
        num_images = min(len(real_images), len(generated_images), 8)
        
        fig, axes = plt.subplots(2, num_images, figsize=(20, 6))
        if num_images == 1:
            axes = axes.reshape(2, 1)
        
        # Plot real images
        for i in range(num_images):
            if real_images[i].max() <= 1.0:
                display_real = real_images[i]
            else:
                display_real = real_images[i] / real_images[i].max()
            
            axes[0, i].imshow(display_real.squeeze(), cmap=self.cmap)
            axes[0, i].set_title(f'Real Ring {i+1}')
            axes[0, i].axis('off')
        
        # Plot generated images
        for i in range(num_images):
            if generated_images[i].max() <= 1.0:
                display_gen = generated_images[i]
            else:
                display_gen = (generated_images[i] + 1) * 0.5
            
            axes[1, i].imshow(display_gen.squeeze(), cmap=self.cmap)
            axes[1, i].set_title(f'Generated Ring {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle('Real vs Generated RICH Cherenkov Rings', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison grid saved: {filename}")