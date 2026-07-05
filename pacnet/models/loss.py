"""
Hybrid Loss Functions for Plant-Adaptive Counting Network.
Combines multiple supervision signals:
1. ℒ_density (MSE): Per-pixel density map regression
2. ℒ_count (MAE): Global count consistency
3. ℒ_local: Local density refinement from TasselNet-style patches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCountingLoss(nn.Module):
    """
    Hybrid loss combining density, counting, and local refinement losses.
    
    ℒ = ℒ_density + λ₁ * ℒ_count + λ₂ * ℒ_local
    """
    def __init__(self, lambda_count=0.1, lambda_local=0.05, lambda_mse=1.0, lambda_local_count=0.2, grid_size=8):
        super().__init__()
        self.lambda_count = lambda_count
        self.lambda_local = lambda_local
        self.lambda_mse = lambda_mse
        self.lambda_local_count = lambda_local_count
        self.grid_size = grid_size

    def forward(self, pred_dict, gt_density, gt_count):
        """Forward pass delegates to multi_scale_loss."""
        return self.multi_scale_loss(pred_dict, gt_density, gt_count)

    def density_loss(self, pred, target):
        """MSE loss on density map with normalization."""
        # Normalize density maps
        pred_sum = pred.sum(dim=(2, 3), keepdim=True) + 1e-8
        target_sum = target.sum(dim=(2, 3), keepdim=True) + 1e-8
        pred_norm = pred / pred_sum
        target_norm = target / target_sum

        # MSE loss on normalized density (focus on distribution)
        mse_norm = F.mse_loss(pred_norm, target_norm)

        # Direct MSE loss (focus on magnitude)
        mse_direct = F.mse_loss(pred, target)

        # MSE weighted by density magnitude
        weight = (target.detach() + 1.0).sqrt()
        mse_weighted = F.mse_loss(pred * weight, target * weight)

        return mse_norm + mse_direct + mse_weighted

    def count_loss(self, pred_density, gt_count):
        """MAE loss on total count."""
        pred_count = pred_density.sum(dim=(2, 3))
        if isinstance(gt_count, torch.Tensor):
            gt_count = gt_count.float().view(-1, 1)
        return F.l1_loss(pred_count, gt_count)

    def local_refine_loss(self, pred_density, gt_density, density_weight=None):
        """
        Local refinement loss focusing on high-density regions.
        Inspired by TasselNet's local counts regression.
        """
        if density_weight is None:
            # Weight based on ground truth density magnitude
            density_weight = (gt_density.detach() + 0.1).sqrt()

        weighted_mse = density_weight * (pred_density - gt_density) ** 2
        return weighted_mse.mean()

    def local_count_loss(self, pred_local_counts, gt_density):
        """
        Local count supervision loss.
        Supervises each grid patch's predicted count against the GT count
        within that patch -- the core TasselNet training signal.

        Parameters
        ----------
        pred_local_counts : Tensor [B, 1, G, G]
            Predicted local patch counts from TasselNetLocalCounter.
        gt_density : Tensor [B, 1, H, W]
            Ground-truth density map.
        """
        G = pred_local_counts.shape[-1]
        B, _, H, W = gt_density.shape
        # Compute GT count per grid cell via spatial integration
        pooled = F.adaptive_avg_pool2d(gt_density, G)
        cell_area = (H * W) / (G * G)
        gt_local = pooled * cell_area
        return F.mse_loss(pred_local_counts, gt_local)

    def multi_scale_loss(self, pred_dict, gt_density, gt_count):
        """
        Multi-scale supervision loss with TasselNet-style local count supervision.

        Loss terms:
        1. L_density:  MSE on reconstructed density map (distribution + magnitude)
        2. L_count:    MAE on global count
        3. L_local:    Weighted MSE on high-density regions (density refinement)
        4. L_patch:    MSE on per-patch local counts (TasselNet core supervision)
        """
        loss = 0.0
        total_loss_detail = {}

        # 1. Main density loss
        loss_density = self.density_loss(pred_dict['density'], gt_density)
        loss += self.lambda_mse * loss_density
        total_loss_detail['density'] = loss_density.item()

        # 2. Auxiliary scale-specific losses (from ScaleAdaptiveHeads)
        target_size = gt_density.shape[-2:]
        for key in ['density_small', 'density_medium', 'density_large']:
            if key in pred_dict and pred_dict[key] is not None:
                pred_scale = pred_dict[key]
                if pred_scale.shape[-2:] != target_size:
                    pred_scale = F.interpolate(
                        pred_scale, size=target_size,
                        mode='bilinear', align_corners=False
                    )
                scale_loss = F.mse_loss(pred_scale, gt_density.detach())
                loss += 0.3 * scale_loss
                total_loss_detail[key] = scale_loss.item()

        # 3. Global count consistency loss
        if gt_count is not None:
            l_count = self.count_loss(pred_dict['density'], gt_count)
            loss += self.lambda_count * l_count
            total_loss_detail['count'] = l_count.item()

        # 4. Local density refinement loss (focus on high-density regions)
        l_local = self.local_refine_loss(pred_dict['density'], gt_density)
        loss += self.lambda_local * l_local
        total_loss_detail['local'] = l_local.item()

        # 5. TasselNet-style local count supervision (NEW)
        if 'local_counts' in pred_dict:
            l_patch = self.local_count_loss(
                pred_dict['local_counts'], gt_density
            )
            loss += self.lambda_local_count * l_patch
            total_loss_detail['local_patch'] = l_patch.item()

        return loss, total_loss_detail

