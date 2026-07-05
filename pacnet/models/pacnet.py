"""
Plant-Adaptive Counting Network (PACNet).
Main model combining:
1. Multi-scale ResNet-FPN backbone (TasselNetV4 inspired)
2. Scale-adaptive density heads with attention fusion
3. TasselNet-style local patch counting (faithful TasselNet implementation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNetFPN, ScaleFeatureFusion
from .multi_scale_head import ScaleAdaptiveHeads
from .local_refine import LocalRefineModule
from .loss import HybridCountingLoss


class PACNet(nn.Module):
    """
    Plant-Adaptive Counting Network.
    
    Architecture:
    Input -> ResNet-50 FPN Backbone -> Scale-Adaptive Density Heads -> TasselNet Local Patches -> Density Map
    
    Key innovations over existing baselines:
    1. Multi-scale feature pyramid with FPN (vs CLIP-Count's single-scale)
    2. Scale-adaptive density heads (vs CountingDINO's fixed exemplar matching)
    3. TasselNet-style grid patch counting (vs CountGD's detection-only pipeline)
    4. Direct local count supervision (patch-wise MSE loss)
    5. Hybrid loss with multi-scale + local count supervision
    """
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        
        # Backbone with FPN for multi-scale features
        self.backbone = ResNetFPN(pretrained=pretrained_backbone)
        
        # Feature fusion across scales
        self.feature_fusion = ScaleFeatureFusion()
        
        # Scale-adaptive density estimation heads
        self.density_heads = ScaleAdaptiveHeads()
        
        # Local refinement module (TasselNet-inspired)
        self.local_refine = LocalRefineModule(feat_dim=256)
        
        # Global context enhancement
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid(),
        )

    def forward(self, x, return_aux=False):
        """
        Args:
            x: Input image tensor [B, 3, H, W]
            return_aux: Return auxiliary outputs for visualization
        Returns:
            density: Final density map [B, 1, H/4, W/4]
            (if return_aux) additional intermediate outputs
        """
        # Extract multi-scale features
        feats = self.backbone(x)  # {'p2','p3','p4','p5'}
        
        # Global context modulation
        context = self.global_context(feats['p2'].mean(dim=(2, 3), keepdim=True))
        for k in feats:
            feats[k] = feats[k] * context[:, :, None, None] + feats[k]
        
        # Scale-adaptive density estimation
        density_out = self.density_heads(feats)
        
        # Local refinement
        fused_feats = self.feature_fusion(feats)
        local_out = self.local_refine(
            fused_feats, global_density=density_out['density']
        )
        
        if return_aux:
            return {
                'density': local_out['density'],
                'local_counts': local_out['local_counts'],
                'total_count': local_out['total_count'],
                'density_raw': density_out['density'],
                'density_small': density_out.get('density_small'),
                'density_medium': density_out.get('density_medium'),
                'density_large': density_out.get('density_large'),
                'scale_weights': density_out.get('scale_weights'),
                'features': feats,
            }
        
        return {'density': local_out['density']}

    def predict(self, x):
        """Convenience method for inference."""
        with torch.no_grad():
            out = self.forward(x, return_aux=False)
            density = out['density']
            count = density.sum(dim=(2, 3))
        return count, density
