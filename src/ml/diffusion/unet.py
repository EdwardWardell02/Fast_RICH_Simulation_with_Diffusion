import numpy as np

class Conv2D:
    """Complete 2D convolution layer with learnable parameters"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier/Glorot initialization for stable training
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.normal(0, scale, (kernel_size, kernel_size, in_channels, out_channels))
        self.bias = np.zeros(out_channels)
        
        # Gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        
    def forward(self, x):
        """Forward pass of convolution"""
        batch_size, h, w, _ = x.shape
        
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.input = x
        self.output = np.zeros((batch_size, out_h, out_w, self.out_channels))
        
        # Perform convolution
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                
                x_slice = x_padded[:, h_start:h_end, w_start:w_end, :]
                
                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        self.output[b, i, j, oc] = np.sum(
                            x_slice[b] * self.weights[:, :, :, oc]
                        ) + self.bias[oc]
        
        return self.output
    
    def backward(self, dout):
        """Backward pass - compute gradients"""
        batch_size, h, w, _ = self.input.shape
        
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0, 0), (self.padding, self.padding), 
                                         (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = self.input
        
        # Initialize gradients
        dx = np.zeros_like(x_padded)
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        
        out_h, out_w = dout.shape[1], dout.shape[2]
        
        # Compute gradients
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.kernel_size
                w_end = w_start + self.kernel_size
                
                for b in range(batch_size):
                    for oc in range(self.out_channels):
                        # Gradient wrt weights
                        self.dweights[:, :, :, oc] += (
                            x_padded[b, h_start:h_end, w_start:w_end, :] * dout[b, i, j, oc]
                        )
                        
                        # Gradient wrt input
                        dx[b, h_start:h_end, w_start:w_end, :] += (
                            self.weights[:, :, :, oc] * dout[b, i, j, oc]
                        )
                
                # Gradient wrt bias
                self.dbias += np.sum(dout[:, i, j, :], axis=0)
        
        # Remove padding from input gradient
        if self.padding > 0:
            dx = dx[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return dx
    
    def update(self, learning_rate):
        """Update weights using gradient descent"""
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias

class UNet:
    """Simplified UNet without complex up/down sampling - easier to debug"""
    
    def __init__(self, image_size=32, channels=1, base_channels=32):
        self.image_size = image_size
        self.channels = channels
        self.base_channels = base_channels
        
        # Simple architecture: 3 convolutional layers
        self.conv1 = Conv2D(channels + base_channels, base_channels * 2)
        self.conv2 = Conv2D(base_channels * 2, base_channels * 4)
        self.conv3 = Conv2D(base_channels * 4, base_channels * 2)
        self.conv4 = Conv2D(base_channels * 2, channels)
        
        # Time embedding
        self.time_embed_dim = base_channels * 4
        self.time_embed_weights = np.random.normal(0, 0.1, (self.time_embed_dim, base_channels))
        
        print(f"Simple U-Net initialized with {self.count_parameters()} parameters")
    
    def count_parameters(self):
        total = 0
        for attr in dir(self):
            if isinstance(getattr(self, attr), Conv2D):
                conv = getattr(self, attr)
                total += np.prod(conv.weights.shape) + np.prod(conv.bias.shape)
        total += np.prod(self.time_embed_weights.shape)
        return total
    
    def _time_embedding(self, t, channels):
        """Time embedding function"""
        half_dim = channels // 2
        emb_coeff = np.log(10000) / (half_dim - 1)
        frequencies = np.exp(np.arange(half_dim, dtype=np.float32) * -emb_coeff)
        
        t_array = np.array(t, dtype=np.float32).reshape(-1, 1)
        angle = t_array * frequencies
        
        sin_emb = np.sin(angle)
        cos_emb = np.cos(angle)
        
        return np.concatenate([sin_emb, cos_emb], axis=-1)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x, t):
        """Simple forward pass"""
        if x.ndim == 3:
            x = x[np.newaxis, ...]
        batch_size = x.shape[0]
        
        # Time embedding
        t_emb = self._time_embedding(t, self.time_embed_dim)
        t_emb = np.dot(t_emb, self.time_embed_weights)
        t_emb = t_emb.reshape(batch_size, 1, 1, -1)
        t_emb_expanded = np.tile(t_emb, (1, x.shape[1], x.shape[2], 1))
        
        # Concatenate with input
        x_with_time = np.concatenate([x, t_emb_expanded], axis=-1)
        
        # Store activations for backward pass
        self.activations = {}
        
        # Forward pass through simple layers
        h1 = self.conv1.forward(x_with_time)
        h1 = self._relu(h1)
        self.activations['h1'] = h1
        
        h2 = self.conv2.forward(h1)
        h2 = self._relu(h2)
        self.activations['h2'] = h2
        
        h3 = self.conv3.forward(h2)
        h3 = self._relu(h3)
        self.activations['h3'] = h3
        
        output = self.conv4.forward(h3)
        self.activations['output'] = output
        
        return output
    
    def backward(self, dout):
        """Simple backward pass"""
        # Layer 4 backward
        dh3 = self.conv4.backward(dout)
        dh3 *= (self.activations['h3'] > 0).astype(np.float32)
        
        # Layer 3 backward
        dh2 = self.conv3.backward(dh3)
        dh2 *= (self.activations['h2'] > 0).astype(np.float32)
        
        # Layer 2 backward
        dh1 = self.conv2.backward(dh2)
        dh1 *= (self.activations['h1'] > 0).astype(np.float32)
        
        # Layer 1 backward
        dx = self.conv1.backward(dh1)
        
        return dx
    
    def update(self, learning_rate):
        """Update all parameters"""
        layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        for layer in layers:
            layer.update(learning_rate)