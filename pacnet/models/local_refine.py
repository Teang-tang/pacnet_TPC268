"""
TasselNet-style Local Patch Counting Module.
A faithful implementation of the core idea from TasselNet (Lu et al., 2017):
local counts regression via regular grid patch division.

Unlike density-map-based methods that predict per-pixel density then integrate,
this module divides the feature map into a regular grid and independently
regresses a scalar count for EACH PATCH. The total count is the SUM of local
patch counts -- each patch is independent, so occlusion in one patch does not
affect others.

Key reference: Lu, H., Cao, Z., Xiao, Y., Zhuang, B., & Shen, C. (2017).
TasselNet: counting maize tassels in the wild via local counts regression
network. Plant Methods, 13(1), 79.

Architecture:
    Fused features [B, C, H, W]
        |
        +--> Feature enhancement (Conv)
        |
        +--> AdaptiveAvgPool2d(grid, grid) -> grid features
        |
        +--> Per-patch MLP -> local_counts [B, 1, G, G]
        |
        +--> Sum -> total_count [B, 1]
        |
        +--> Upsample + fuse with global density -> density [B, 1, H, W]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRefineModule(nn.Module):
    """
    TasselNet-style Local Patch Counting Module.

    Faithfully implements the core insight from TasselNet (2017):
    instead of a global density integral, predict independent local counts
    for a regular grid of patches and sum them.

    Key advantage for plant counting: in dense plant organs, per-pixel
    density maps suffer from feature overlap -> systematic underestimation.
    Per-patch independent counting avoids this issue because each patch
    only needs to decide "how many objects are in THIS region" rather
    than "where exactly is each object."

    Parameters
    ----------
    feat_dim : int
        Number of input feature channels (default: 256).
    grid_size : int
        Number of patches along each spatial dimension (default: 8).
        Total patches = grid_size x grid_size.
    """
    def __init__(self, feat_dim=256, grid_size=8):
        super().__init__()
        self.grid_size = grid_size

        # Feature enhancement before patch splitting
        self.pre_conv = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Per-patch independent counting head
        self.patch_counter = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        # Density map reconstructor
        self.density_fuser = nn.Sequential(
            nn.Conv2d(feat_dim + 1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    @staticmethod
    def compute_gt_local_counts(gt_density, grid_size):
        """
        Compute ground-truth local counts from a density map.

        For each grid cell, the GT local count = spatial integral of
        the density map within that cell.
        """
        B, _, H, W = gt_density.shape
        pooled = F.adaptive_avg_pool2d(gt_density, grid_size)
        cell_area = (H * W) / (grid_size * grid_size)
        return pooled * cell_area

    def forward(self, feats, global_density=None):
        """
        Parameters
        ----------
        feats : Tensor [B, C, H, W] -- fused multi-scale feature map.
        global_density : Tensor [B, 1, H, W] or None
            Optional global density map for density reconstruction.

        Returns
        -------
        dict with keys:
            - local_counts : [B, 1, G, G]  per-patch independent counts
            - total_count  : [B, 1]         sum of local counts
            - density      : [B, 1, H, W]   reconstructed density map
        """
        B, C, H, W = feats.shape

        # Step 1: Feature enhancement
        enhanced = self.pre_conv(feats)                       # [B, 64, H, W]

        # Step 2: Grid pooling -> each cell = one patch
        grid_feats = F.adaptive_avg_pool2d(
            enhanced, self.grid_size
        )                                                     # [B, 64, G, G]

        # Step 3: Per-patch independent count regression
        local_counts = F.relu(
            self.patch_counter(grid_feats)
        )                                                     # [B, 1, G, G]

        # Step 4: Total count = sum of independent patch counts
        total_count = local_counts.sum(dim=(1, 2, 3), keepdim=True)  # [B, 1]

        # Step 5: Reconstruct density map (for loss & visualization)
        coarse_density = F.interpolate(
            local_counts, size=(H, W), mode='bilinear', align_corners=False
        )                                                     # [B, 1, H, W]

        if global_density is not None:
            cat = torch.cat([feats, coarse_density], dim=1)
            density = self.density_fuser(cat)
            density = F.relu(density)
            # Renormalize to preserve total count from local patches
            density = density * total_count / (
                density.sum(dim=(1, 2, 3), keepdim=True) + 1e-8
            )
        else:
            density = coarse_density

        return {
            'local_counts': local_counts,
            'total_count': total_count,
            'density': density,
        }