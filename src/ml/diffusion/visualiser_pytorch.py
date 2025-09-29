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