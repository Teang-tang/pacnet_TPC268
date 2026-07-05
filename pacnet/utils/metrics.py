"""
Evaluation Metrics for Plant Counting.
MAE, RMSE, R², rMAE, rRMSE, FLOPS estimation.
Same metrics as the three baselines for fair comparison.
"""
import numpy as np
import torch


def compute_metrics(pred_counts, gt_counts):
    """
    Compute all counting metrics.
    Args:
        pred_counts: list/array of predicted counts
        gt_counts: list/array of ground truth counts
    Returns:
        dict with MAE, RMSE, R2, rMAE, rRMSE, Mean_GT, Mean_Pred
    """
    preds = np.array(pred_counts, dtype=np.float64)
    gts = np.array(gt_counts, dtype=np.float64)
    n = len(preds)

    if n == 0:
        return {}

    # Absolute errors
    errors = preds - gts
    abs_errors = np.abs(errors)

    # MAE: Mean Absolute Error
    mae = float(np.mean(abs_errors))

    # RMSE: Root Mean Squared Error
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # R²: Coefficient of determination
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((gts - np.mean(gts)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    # Relative metrics
    mean_gt = float(np.mean(gts))
    mean_pred = float(np.mean(preds))
    rmae = mae / (mean_gt + 1e-10)
    rrmse = rmse / (mean_gt + 1e-10)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'rMAE': rmae,
        'rRMSE': rrmse,
        'Mean_GT': mean_gt,
        'Mean_Pred': mean_pred,
        'Samples': n,
    }


def estimate_flops(model, input_size=(1, 3, 384, 384)):
    """Estimate FLOPS using torchprofile or manual estimation."""
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        dummy = torch.randn(*input_size).to(device)
        flops = FlopCountAnalysis(model, dummy).total()
        return flops / 1e9  # GFLOPs
    except ImportError:
        # Manual estimation
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Rough FLOPs estimate: 2 * params per forward pass (approximate)
        flops = 2 * params * input_size[-1] * input_size[-2] / (16 * 16)  # density map resolution factor
        return flops / 1e9


def print_results(results_dict, model_name='PACNet', flops=None):
    """Print evaluation results in a formatted table."""
    print(f'\n{"="*50}')
    print(f'{model_name} on TPC-268 ({results_dict.get("Samples", "?")} images)')
    print(f'{"="*50}')
    print(f'  MAE:     {results_dict["MAE"]:.4f}')
    print(f'  RMSE:    {results_dict["RMSE"]:.4f}')
    print(f'  R2:      {results_dict["R2"]:.4f}')
    print(f'  rMAE:    {results_dict["rMAE"]:.4f}')
    print(f'  rRMSE:   {results_dict["rRMSE"]:.4f}')
    print(f'  Mean_GT: {results_dict["Mean_GT"]:.4f}')
    print(f'  Mean_Pred: {results_dict["Mean_Pred"]:.4f}')
    if flops:
        print(f'  FLOPS:   {flops:.2f} GFLOPs')
    print(f'{"="*50}')
