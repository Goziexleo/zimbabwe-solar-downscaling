"""Canonical CNN spatial super-resolution architecture (Section 3.6.5).

Shared by train_cnn_downscaler.py and evaluate_cnn.py so the architecture
can never drift from the trained checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperResolutionCNN(nn.Module):
    """Pre-upsampling bilinear interpolation to the fine 0.1 deg grid,
    followed by three 64-filter conv blocks and a 1x1 ReLU output layer."""
    def __init__(self, in_channels, fine_shape):
        super().__init__()
        self.fine_shape = tuple(fine_shape)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_conv = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x_upsampled = F.interpolate(x, size=self.fine_shape, mode='bilinear', align_corners=False)
        out = self.block1(x_upsampled)
        out = self.block2(out)
        out = self.block3(out)
        return torch.clamp(self.out_conv(out), 0.0, 1.1)


def load_cnn_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SuperResolutionCNN(
        in_channels=len(checkpoint['feature_vars']), fine_shape=checkpoint['fine_shape'],
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
