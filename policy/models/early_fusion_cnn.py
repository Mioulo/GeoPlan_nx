"""Convolutional early fusion of two image feature maps and pose channels."""
import torch
from torch import nn


class EarlyFusionCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Relative pose is [qx,qy,qz,qw,tx,ty,tz], broadcast spatially.
        # Multiples of 32 keep the 128-wide student's fusion capacity close to
        # the existing ResNet + 2-frame/2-fusion-layer Transformer (1.09 M).
        first_channels, second_channels = 192, 224
        self.layers = nn.Sequential(
            nn.Conv2d(2 * channels + 7, first_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, first_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(first_channels, second_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, second_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(second_channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, features, relative_pose):
        if features.ndim != 5 or features.shape[1] != 2:
            raise ValueError('Early-fusion CNN requires two feature maps [B,2,C,H,W]')
        batch, _, channels, height, width = features.shape
        if relative_pose is None or relative_pose.shape != (batch, 7):
            raise ValueError('Early-fusion CNN requires relative pose [B,7]')
        pose_channels = relative_pose.to(device=features.device, dtype=features.dtype)
        pose_channels = pose_channels[:, :, None, None].expand(-1, -1, height, width)
        fused_input = torch.cat((features.reshape(batch, 2 * channels, height, width), pose_channels), dim=1)
        return self.layers(fused_input)
