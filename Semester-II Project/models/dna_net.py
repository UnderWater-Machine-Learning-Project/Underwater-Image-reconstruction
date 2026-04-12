
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Encapsulates Convolution -> Batch Normalization -> Activation"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            # padding=1 ensures the spatial dimensions don't shrink during convolution
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # BatchNorm normalizes the outputs of the layer, speeding up training
            nn.BatchNorm2d(out_ch),
            # ReLU (Rectified Linear Unit) introduces non-linearity (f(x) = max(0, x))
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DNANet(nn.Module):
    def __init__(self):
        super(DNANet, self).__init__()
        
        # ENCODER: Extracting features
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2) # Downsamples spatial dimensions by 2
        
        # DECODER: Reconstructing the clear image
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input is 64 (from upsample) + 32 (skip connection from enc1)
        self.dec1 = ConvBlock(96, 32) 
        
        # FINAL LAYER: Map back to 3 channels (RGB)
        self.final = nn.Conv2d(32, 3, kernel_size=1)
        # Sigmoid squashes the final math output strictly to the [0.0, 1.0] range
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass logic with a Skip Connection
        x1 = self.enc1(x)               # High-res features
        x2 = self.enc2(self.pool(x1))   # Low-res, deep features
        
        x_up = self.up(x2)              # Scale back up
        x_merge = torch.cat([x_up, x1], dim=1)  # Combine deep features with high-res details
        
        d1 = self.dec1(x_merge)
        return self.sigmoid(self.final(d1))