"""
Multi-Scale Density Heads with Adaptive Scale Selection.
Key innovation: Multiple density heads at different scales,
with learned per-image scale attention weights.
This addresses the 3-4 orders of magnitude scale variation in plant images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityHead(nn.Module):
    """Single density estimation head at a specific feature scale."""
    def __init__(self, in_channels=256, hidden_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.head(x)


class ScaleAdaptiveHeads(nn.Module):
    """
    Scale-Adaptive Density Estimation Module.
    Uses 3 parallel density heads at different feature scales,
    with learnable attention weights to adaptively combine them per image.
    
    Small head: P2 (1/4) - for small objects (stomata, seeds)
    Medium head: P3 (1/8) - for medium objects (flowers, leaves)
    Large head: P4 (1/16) - for large objects (fruits, whole plants)
    """
    def __init__(self, feat_channels=256):
        super().__init__()
        # Three scale-specific density heads
        self.small_head = DensityHead(feat_channels)   # P2: fine-grained
        self.medium_head = DensityHead(feat_channels)  # P3: mid-scale
        self.large_head = DensityHead(feat_channels)   # P4: coarse

        # Global context for scale attention
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channels * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(dim=1),
        )

        # Target size (P2: 1/4 of input)
        self.target_scale = 'p2'

    def forward(self, feats_dict):
        p2 = feats_dict['p2']  # 1/4
        p3 = feats_dict['p3']  # 1/8
        p4 = feats_dict['p4']  # 1/16

        # Generate density maps at each scale
        d_small = self.small_head(p2)
        d_medium = self.medium_head(p3)
        d_large = self.large_head(p4)

        # Upsample medium and large to match small resolution
        target_size = d_small.shape[-2:]
        d_medium_up = F.interpolate(d_medium, size=target_size, mode='bilinear', align_corners=False)
        d_large_up = F.interpolate(d_large, size=target_size, mode='bilinear', align_corners=False)

        # Compute scale attention weights from global features
        # Pool features from all scales
        pooled = torch.cat([
            F.adaptive_avg_pool2d(p2, 1),
            F.adaptive_avg_pool2d(p3, 1),
            F.adaptive_avg_pool2d(p4, 1),
        ], dim=1)
        weights = self.scale_attention(pooled)  # [B, 3]

        # Attention-weighted fusion
        w_s = weights[:, 0:1, None, None]
        w_m = weights[:, 1:2, None, None]
        w_l = weights[:, 2:3, None, None]

        fused = w_s * d_small + w_m * d_medium_up + w_l * d_large_up

        return {
            'density': fused,
            'density_small': d_small,
            'density_medium': d_medium,
            'density_large': d_large,
            'scale_weights': weights,
        }
