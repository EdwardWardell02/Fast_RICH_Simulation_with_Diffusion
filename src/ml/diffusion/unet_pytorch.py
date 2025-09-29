# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding as used in original DDPM paper"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        # Convert timestep to sinusoidal embedding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    """Basic residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Time embedding projection
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Shortcut connection if channel dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.conv1(x)
        
        # Add time embedding
        t_emb = self.time_emb(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        h = h + t_emb
        
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Self-attention block for better long-range dependencies"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)  # B, N, C
        k = self.k(h).view(B, C, -1)  # B, C, N
        v = self.v(h).view(B, C, -1).permute(0, 2, 1)  # B, N, C
        
        # Attention mechanism
        attention = torch.bmm(q, k) * (C ** -0.5)
        attention = F.softmax(attention, dim=-1)
        
        h = torch.bmm(attention, v)  # B, N, C
        h = h.permute(0, 2, 1).view(B, C, H, W)
        h = self.proj_out(h)
        
        return x + h

class Downsample(nn.Module):
    """Downsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """Upsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class PyTorchUNet(nn.Module):
    """
    PyTorch UNet implementation for DDPM
    Based on the original DDPM architecture with improvements from guided-diffusion
    """
    
    def __init__(self, image_size=32, in_channels=1, base_channels=32, 
                 channel_mults=(1, 2, 4, 8), num_res_blocks=2, attention_resolutions=(16,),
                 dropout=0.1, time_emb_dim=128):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Time embedding
        self.time_emb = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsample stages
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        
        # Create downsampling path
        downsample_channel_mults = [1] + list(channel_mults)
        for i, mult in enumerate(downsample_channel_mults[1:]):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(
                    now_channels, out_channels, time_emb_dim, dropout
                ))
                now_channels = out_channels
                channels.append(now_channels)
                
                # Add attention at specified resolutions
                if image_size in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(now_channels))
            
            if i != len(downsample_channel_mults[1:]) - 1:  # Don't downsample at last stage
                self.down_blocks.append(Downsample(now_channels))
                channels.append(now_channels)
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        ])
        
        # Create upsampling path (reverse of downsampling)
        for i, mult in enumerate(reversed(downsample_channel_mults[1:])):
            out_channels = base_channels * mult
            for j in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(
                    channels.pop() + now_channels, out_channels, time_emb_dim, dropout
                ))
                now_channels = out_channels
                
                # Add attention at specified resolutions
                if image_size in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(now_channels))
            
            if i != len(downsample_channel_mults[1:]) - 1:  # Don't upsample at last stage
                self.up_blocks.append(Upsample(now_channels))
        
        # Final layers
        self.final_norm = nn.GroupNorm(32, now_channels)
        self.final_conv = nn.Conv2d(now_channels, in_channels, kernel_size=3, padding=1)
        
        print(f"PyTorch UNet initialized with {self.count_parameters():,} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, t):
        """
        Forward pass of the UNet
        x: input tensor of shape (batch, channels, height, width)
        t: timestep tensor of shape (batch,)
        """
        # Ensure proper input dimensions
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Time embedding
        t_emb = self.time_emb(t)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Downsample path
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
                skip_connections.append(h)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # Downsample
                h = module(h)
                skip_connections.append(h)
        
        # Middle path
        for module in self.middle_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:  # AttentionBlock
                h = module(h)
        
        # Upsample path
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                # Concatenate with skip connection
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # Upsample
                h = module(h)
        
        # Final layers
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)
        
        return h

# Alias for backward compatibility
UNet = PyTorchUNet