"""
Visualization utilities for counting results.
Same format as the three baselines for consistent comparison.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def save_scatter_plot(gt_counts, pred_counts, save_path, title='PACNet'):
    """Scatter plot of predicted vs ground truth counts."""
    plt.figure(figsize=(8, 6))
    plt.scatter(gt_counts, pred_counts, alpha=0.5, s=20, c='darkorange')
    lo, hi = 0, max(max(gt_counts), max(pred_counts), 1) * 1.05
    plt.plot([lo, hi], [lo, hi], 'r--', lw=1.5, alpha=0.7)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Ground Truth', fontsize=12)
    plt.ylabel('Predicted Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_error_histogram(errors, save_path, mae=None):
    """Error distribution histogram."""
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=40, alpha=0.7, color='darkorange', edgecolor='white')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
    plt.xlabel('Error (Pred - GT)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    title = 'Error Distribution'
    if mae:
        title += f' (MAE={mae:.2f})'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_example_predictions(samples, preds, gts, save_path, n_examples=5):
    """Best and worst prediction examples."""
    abs_errors = np.abs(np.array(preds) - np.array(gts))
    n = min(n_examples, len(preds))
    best_idx = np.argsort(abs_errors)[:n]
    worst_idx = np.argsort(-abs_errors)[:n]

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for row, idx_list in enumerate([best_idx, worst_idx]):
        for col, idx in enumerate(idx_list):
            ax = axes[row, col]
            ax.axis('off')
            try:
                img = Image.open(samples[idx]['path']).convert('RGB')
                ax.imshow(img)
            except:
                pass
            ax.set_title(
                f'GT={gts[idx]:.0f}, Pred={preds[idx]:.0f}\n'
                f'{samples[idx].get("organization","")}',
                fontsize=8
            )

    plt.suptitle('Best Predictions (top) and Worst Predictions (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_density_visualization(image, density_map, save_path, gt=None, pred=None):
    """Visualize image with density map overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(image)
    ax1.set_title(f'Input Image (GT={gt}, Pred={pred})' if gt else 'Input Image')
    ax1.axis('off')

    density_display = np.maximum(density_map.squeeze(), 0)
    im = ax2.imshow(density_display, cmap='hot', interpolation='nearest')
    ax2.set_title(f'Density Map (sum={pred:.1f})' if pred else 'Density Map')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_scale_attention_visualization(weights, save_path):
    """Visualize scale attention weights across the dataset."""
    plt.figure(figsize=(8, 4))
    labels = ['Small (fine)', 'Medium', 'Large (coarse)']
    x = np.arange(len(labels))
    mean_weights = np.mean(weights, axis=0) if len(np.array(weights).shape) > 1 else weights

    plt.bar(x, mean_weights, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Feature Scale')
    plt.ylabel('Attention Weight')
    plt.title('Scale Attention Distribution')
    plt.xticks(x, labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
