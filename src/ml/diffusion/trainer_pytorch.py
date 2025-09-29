import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import os

class RICHTrainer:
    """Complete PyTorch trainer for RICH DDPM"""
    
    def __init__(self, ddpm, unet, visualizer, learning_rate=1e-4, device='auto'):
        self.ddpm = ddpm
        self.unet = unet
        self.visualizer = visualizer
        self.learning_rate = learning_rate
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device
        self.unet = self.unet.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.losses = []
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Learning rate: {learning_rate}")
    
    def training_step(self, x0_batch, t, true_noise):
        """
        Single training step - forward pass, loss computation, backward pass
        """
        # Move data to device
        x0_batch = x0_batch.to(self.device)
        t = t.to(self.device)
        true_noise = true_noise.to(self.device)
        
        # Forward diffusion to get noisy images
        xt, _ = self.ddpm.forward_diffusion(x0_batch, t, true_noise)
        
        # Predict noise using UNet
        predicted_noise = self.unet(xt, t)
        
        # Compute loss
        loss = self.loss_fn(predicted_noise, true_noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, normalised_images, epochs=100, batch_size=16, validation_split=0.1):
        """
        Complete training loop with validation
        """
        print(f"Starting training on {len(normalised_images)} images")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Convert numpy arrays to PyTorch tensors
        images_tensor = torch.FloatTensor(normalised_images)
        
        # Move channel dimension for PyTorch: (N, H, W, C) -> (N, C, H, W)
        if images_tensor.dim() == 4 and images_tensor.shape[-1] in [1, 3]:
            images_tensor = images_tensor.permute(0, 3, 1, 2)
        
        # Create dataset and dataloader
        dataset = TensorDataset(images_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                               num_workers=0, pin_memory=True)
        
        # Calculate steps per epoch
        steps_per_epoch = len(dataloader)
        print(f"Steps per epoch: {steps_per_epoch}")
        
        for epoch in range(epochs):
            self.unet.train()  # Set model to training mode
            epoch_losses = []
            
            for batch_idx, (batch_images,) in enumerate(dataloader):
                # Move batch to device
                x0_batch = batch_images.to(self.device)
                batch_size_current = x0_batch.shape[0]
                
                # Sample random timesteps and noise
                t = torch.randint(0, self.ddpm.timesteps, (batch_size_current,), 
                                 device=self.device)
                true_noise = torch.randn_like(x0_batch, device=self.device)
                
                # Perform training step
                loss = self.training_step(x0_batch, t, true_noise)
                epoch_losses.append(loss)
                
                # Print progress
                if batch_idx % max(1, steps_per_epoch // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{steps_per_epoch}, "
                          f"Loss: {loss:.6f}")
            
            # Calculate average epoch loss
            avg_loss = np.mean(epoch_losses)
            self.losses.append(avg_loss)
            
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch + 1)
                self._generate_sample(epoch + 1)
        
        # Plot training progress
        self.visualizer.plot_training_progress(self.losses)
        print("Training completed successfully!")
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'learning_rate': self.learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def _generate_sample(self, epoch):
        """Generate sample during training"""
        print(f"Generating sample at epoch {epoch}...")
        
        # Set model to evaluation mode
        self.unet.eval()
        
        with torch.no_grad():
            # Generate samples using PyTorch tensors
            final_samples, intermediate_samples = self.ddpm.generate_samples(
                self.unet, num_samples=1, device=self.device
            )
            
            # Convert to numpy for visualization
            final_samples_np = final_samples.cpu().numpy()
            intermediate_samples_np = [sample.cpu().numpy() for sample in intermediate_samples]
            
            # Plot reverse process
            timesteps = list(range(self.ddpm.timesteps - 1, -1, -200))
            self.visualizer.plot_reverse_process(
                self.ddpm,           # The DDPM model should be the first argument
                self.unet,           # The UNet model is the second argument
                num_samples=1,       # Specify number of samples
                num_steps=8,         # Specify number of visualization steps
                filename=f"sample_epoch_{epoch}.png"
            )
        
        # Set back to training mode
        self.unet.train()

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']
        print(f"Checkpoint loaded from {checkpoint_path}")