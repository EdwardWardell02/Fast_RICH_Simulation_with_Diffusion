import numpy as np
from datetime import datetime
import os

class RICHTrainer:
    """Complete training class with gradient clipping"""
    
    def __init__(self, ddpm, unet, visualizer, learning_rate=1e-4, max_gradient_norm=1.0):
        self.ddpm = ddpm
        self.unet = unet
        self.visualizer = visualizer
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm  # Gradient clipping threshold
        self.losses = []
        
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def compute_gradients(self, x0_batch, t, true_noise):
        """
        Compute gradients with dimension handling
        """
        batch_size = x0_batch.shape[0]
        
        # Ensure input has correct dimensions
        if x0_batch.ndim == 3:
            x0_batch = x0_batch[np.newaxis, ...]
        
        # Forward pass through DDPM to get noisy images
        xt, _ = self.ddpm.forward_diffusion(x0_batch, t, true_noise)
        
        # Ensure xt has correct shape
        if xt.ndim == 3:
            xt = xt[np.newaxis, ...]
        
        # Forward pass through U-Net to predict noise
        predicted_noise = self.unet.forward(xt, t)
        
        # Compute loss and gradient
        loss = np.mean((true_noise - predicted_noise) ** 2)
        
        # Backward pass
        d_predicted_noise = 2 * (predicted_noise - true_noise) / batch_size
        
        # Backpropagate through U-Net
        self.unet.backward(d_predicted_noise)
        
        return loss
    
    def _clip_gradients(self, max_norm=None):
        """Clip gradients to prevent explosion"""
        if max_norm is None:
            max_norm = self.max_gradient_norm
            
        total_norm = 0.0
        layers = self._get_all_layers()
        
        # Calculate total gradient norm
        for layer in layers:
            if hasattr(layer, 'dweights'):
                total_norm += np.sum(layer.dweights ** 2)
            if hasattr(layer, 'dbias'):
                total_norm += np.sum(layer.dbias ** 2)
        
        total_norm = np.sqrt(total_norm)
        
        # Clip if norm exceeds threshold
        if total_norm > max_norm:
            clip_factor = max_norm / (total_norm + 1e-10)
            for layer in layers:
                if hasattr(layer, 'dweights'):
                    layer.dweights *= clip_factor
                if hasattr(layer, 'dbias'):
                    layer.dbias *= clip_factor
            print(f"Gradients clipped: norm {total_norm:.3f} -> {max_norm:.3f}")
            return True  # Clipping was applied
        return False  # No clipping needed
    
    def train(self, normalised_images, epochs=100, batch_size=16):
        """
        Training loop with gradient clipping and explosion detection
        """
        print(f"Starting training on {len(normalised_images)} images")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {self.learning_rate}")
        print(f"Gradient clipping enabled: max_norm={self.max_gradient_norm}")
        
        # Handle small datasets
        if batch_size > len(normalised_images):
            batch_size = max(1, len(normalised_images))
            print(f"Reduced batch size to {batch_size} for small dataset")
        
        num_batches = max(1, len(normalised_images) // batch_size)
        explosion_count = 0
        max_explosions = 5  # Stop if we get too many explosions
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data each epoch
            indices = np.random.permutation(len(normalised_images))
            shuffled_data = normalised_images[indices]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(normalised_images))
                
                if start_idx >= end_idx:
                    continue
                    
                x0_batch = shuffled_data[start_idx:end_idx]
                actual_batch_size = len(x0_batch)
                
                # Sample random timesteps and noise
                t = np.random.randint(0, self.ddpm.timesteps, (actual_batch_size,))
                true_noise = np.random.normal(0, 1, x0_batch.shape)
                
                try:
                    # Compute gradients and loss
                    loss = self.compute_gradients(x0_batch, t, true_noise)
                    
                    # Apply gradient clipping
                    clipped = self._clip_gradients()
                    
                    # Check for explosion
                    if np.isnan(loss) or np.isinf(loss) or loss > 1e6:
                        explosion_count += 1
                        print(f"🚨 Explosion detected! Loss: {loss:.2e}, Count: {explosion_count}/{max_explosions}")
                        
                        if explosion_count >= max_explosions:
                            print("Too many explosions, stopping training.")
                            return
                        
                        # Reset gradients and skip update
                        self._zero_gradients()
                        continue
                    
                    # Update weights using gradient descent
                    self.unet.update(self.learning_rate)
                    
                    epoch_losses.append(loss)
                    
                    # Print progress more frequently for small datasets
                    print_interval = max(1, num_batches // 10)  # Print 10 times per epoch
                    if batch_idx % print_interval == 0:
                        clip_status = "[CLIPPED]" if clipped else ""
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, "
                              f"Loss: {loss:.6f} {clip_status}")
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    explosion_count += 1
                    if explosion_count >= max_explosions:
                        print("Too many errors, stopping training.")
                        return
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                self.losses.append(avg_loss)
                print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
            else:
                print(f"Epoch {epoch+1} completed. No successful batches.")
            
            # Save checkpoint more frequently for small datasets
            if (epoch + 1) % max(1, epochs // 5) == 0:  # Save 5 checkpoints total
                self._save_checkpoint(epoch + 1)
                if len(normalised_images) > 10:  # Only generate samples if we have enough data
                    self._generate_sample(epoch + 1)
        
        # Final visualizations
        if self.losses:
            self.visualizer.plot_training_progress(self.losses)
        print("Training completed!")
    
    def _zero_gradients(self):
        """Reset gradients to zero after explosion"""
        for layer in self._get_all_layers():
            if hasattr(layer, 'dweights'):
                layer.dweights = np.zeros_like(layer.dweights)
            if hasattr(layer, 'dbias'):
                layer.dbias = np.zeros_like(layer.dbias)
    
    def _get_all_layers(self):
        """Get all trainable layers from U-Net"""
        layers = []
        # Add all convolutional layers
        for attr_name in dir(self.unet):
            attr = getattr(self.unet, attr_name)
            if hasattr(attr, 'dweights') or hasattr(attr, 'dbias'):
                layers.append(attr)
        return layers
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'losses': self.losses,
                'learning_rate': self.learning_rate,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save weights for each layer
            for i, layer in enumerate(self._get_all_layers()):
                if hasattr(layer, 'weights'):
                    checkpoint[f'layer_{i}_weights'] = layer.weights
                if hasattr(layer, 'bias'):
                    checkpoint[f'layer_{i}_bias'] = layer.bias
            
            filename = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.npz"
            np.savez(filename, **checkpoint)
            print(f"Checkpoint saved: {filename}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def _generate_sample(self, epoch):
        """Generate sample during training"""
        try:
            print(f"Generating sample at epoch {epoch}...")
            final_sample, intermediate_samples = self.ddpm.generate_samples(
                self.unet, num_samples=1
            )
            # Save sample visualization...
        except Exception as e:
            print(f"Error generating sample: {e}")