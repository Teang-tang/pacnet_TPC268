"""
密度图生成工具 —— 从 TPC-268 点标注生成自适应高斯密度图
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def generate_density_map(points, img_size, out_size=None, sigma=15, adaptive=True):
    """
    从点标注生成密度图。
    Args:
        points: [(x1,y1), ...] 或 (N,2) numpy 数组，点坐标
        img_size: (H, W) 原始图像尺寸
        out_size: (h, w) 输出密度图尺寸（默认 img_size/4）
        sigma: 高斯核标准差
        adaptive: 是否使用自适应高斯核（基于最近邻距离）
    Returns:
        density: (out_h, out_w) numpy 密度图
    """
    H, W = img_size
    if out_size is None:
        out_h, out_w = H // 4, W // 4
    else:
        out_h, out_w = out_size

    points = np.array(points, dtype=np.float64)
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((out_h, out_w), dtype=np.float64)

    # 在原始分辨率上创建密度图
    density = np.zeros((H, W), dtype=np.float64)

    if adaptive and len(points) > 1:
        # 自适应高斯核：KNN 距离决定每个点的 sigma
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        for i, pt in enumerate(points):
            d = np.sort(dists[i])
            nearest_dist = d[1] if len(d) > 1 else 15.0  # 到最近邻的距离
            adapt_sigma = max(min(nearest_dist / 2.0, 30.0), 3.0)
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < W and 0 <= y < H:
                density[y, x] = 1.0
                # 用自适应 sigma 的高斯核平滑
                # 但这里先全部放点，再统一高斯滤波
    else:
        for pt in points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < W and 0 <= y < H:
                density[y, x] = 1.0

    # 对原始高分辨率密度图做高斯滤波
    density = gaussian_filter(density, sigma=sigma if not adaptive else None)
    if adaptive:
        # 如果使用自适应，用中位数距离作为 sigma
        if len(points) > 1:
            from scipy.spatial.distance import cdist
            dists = cdist(points, points)
            min_dists = [np.sort(dists[i])[1] if len(dists[i]) > 1 else 15.0 for i in range(len(points))]
            med_dist = np.median(min_dists)
            sigma = max(min(med_dist / 2.0, 20.0), 3.0)
        density = gaussian_filter(density, sigma=sigma)

    # 归一化：使得 sum = 目标总数
    scale = len(points) / (density.sum() + 1e-10)
    density *= scale

    # 降采样到输出尺寸
    from scipy.ndimage import zoom
    scale_h, scale_w = out_h / H, out_w / W
    density = zoom(density, (scale_h, scale_w), order=1)

    # 重新归一化
    scale = len(points) / (density.sum() + 1e-10)
    density *= scale

    return density.astype(np.float64)


def generate_density_map_fast(points, H, W, out_h, out_w):
    """快速版本：直接在输出分辨率上生成密度图（不经过高分辨率中间步骤）"""
    points = np.array(points, dtype=np.float64)
    if len(points.shape) == 1: points = points.reshape(-1, 2)
    if len(points) == 0: return np.zeros((out_h, out_w), dtype=np.float32)

    # 缩放到输出分辨率
    pts = points.copy()
    pts[:, 0] = pts[:, 0] * out_w / W
    pts[:, 1] = pts[:, 1] * out_h / H

    # 在输出分辨率上创建密度图
    density = np.zeros((out_h, out_w), dtype=np.float64)

    # 自适应 sigma
    from scipy.spatial.distance import cdist
    if len(points) > 1:
        dists = cdist(pts, pts)
        min_dists = [np.sort(dists[i])[1] if len(dists[i]) > 1 else 3.0 for i in range(len(points))]
        med_sigma = np.median(min_dists)
        sigma = max(min(med_sigma, 5.0), 1.0)
    else:
        sigma = 3.0

    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < out_w and 0 <= y < out_h:
            density[y, x] = 1.0

    density = gaussian_filter(density, sigma=sigma)
    scale = len(points) / (density.sum() + 1e-10)
    density *= scale
    return density.astype(np.float32)


def generate_density_map_torch(points, H, W, out_h, out_w, device="cpu"):
    """PyTorch 版本，支持 batch 处理"""
    pts = np.array(points, dtype=np.float64)
    if len(pts.shape) == 1: pts = pts.reshape(-1, 2)

    density = np.zeros((out_h, out_w), dtype=np.float32)
    if len(pts) == 0: return torch.from_numpy(density)

    # 缩放到输出分辨率
    pts[:, 0] = pts[:, 0] * out_w / max(W, 1)
    pts[:, 1] = pts[:, 1] * out_h / max(H, 1)

    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < out_w and 0 <= y < out_h:
            density[y, x] += 1.0

    # 高斯平滑
    from scipy.ndimage import gaussian_filter
    sigma = max(min(np.median(np.diff(pts, axis=0)), 5.0) if len(pts) > 1 else 3.0, 1.0)
    density = gaussian_filter(density, sigma=sigma)

    # 归一化
    s = len(pts) / (density.sum() + 1e-10)
    density *= s
    return torch.from_numpy(density)


def generate_density_batch(points_list, img_sizes, out_h, out_w):
    """批量生成密度图供训练使用"""
    densities = []
    for points, (H, W) in zip(points_list, img_sizes):
        d = generate_density_map_fast(points, H, W, out_h, out_w)
        densities.append(torch.from_numpy(d).unsqueeze(0))
    return torch.stack(densities)
