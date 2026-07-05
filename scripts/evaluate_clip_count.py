"""
CLIP-Count Evaluation on TPC-268 Dataset

Usage: python scripts/evaluate_clip_count.py [--num_images 100] [--output_dir results/clip_count]

This script:
1. Loads the CLIP-Count model with official pretrained weights
2. Loads TPC-268 dataset (random subset)
3. Runs text-guided density map inference
4. Evaluates with MAE, RMSE, R2, rMAE, rRMSE
5. Generates visualizations
"""
import sys, os, types, json, math, random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

# ===== COMPATIBILITY PATCHES =====
torch_six = types.ModuleType('torch._six')
torch_six.inf = float('inf')
sys.modules['torch._six'] = torch_six
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'float': [np.float16, np.float32, np.float64],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'complex': [np.complex64, np.complex128],
    }
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ===== PROJECT SETUP =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'baselines' / 'clip_count'))
TPC_DATA_DIR = str(PROJECT_ROOT / 'TPC-268' / 'TPC-268')
TPC_SPLITS = str(PROJECT_ROOT / 'TPC-268-main' / 'splits')
TPC_ANNO = str(PROJECT_ROOT / 'TPC-268-main' / 'annotations' / 'tpc268_annotations.json')
WEIGHTS_DIR = str(PROJECT_ROOT / 'weights')
OUTPUT_DIR = str(PROJECT_ROOT / 'results' / 'clip_count')

from run import Model, get_args_parser
from util.constant import SCALE_FACTOR


def compute_metrics(pred_counts, gt_counts):
    preds = np.array(pred_counts)
    gts = np.array(gt_counts)
    n = len(preds)
    if n == 0:
        return {}
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    ss_res = np.sum((preds - gts) ** 2)
    ss_tot = np.sum((gts - np.mean(gts)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    mean_gt = float(np.mean(gts))
    rmae = mae / (mean_gt + 1e-10)
    rrmse = rmse / (mean_gt + 1e-10)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'rMAE': rmae, 'rRMSE': rrmse, 'Mean_GT': mean_gt, 'Mean_Pred': float(np.mean(preds))}


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    W, H = image.size
    new_H = 16 * int(H / 16)
    new_W = 16 * int(W / 16)
    image = transforms.Resize((new_H, new_W))(image)
    tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0)


def load_model(ckpt_path, device='cuda'):
    parser = get_args_parser()
    args = parser.parse_args([])
    args.use_contrast = False
    args.batch_size = 1
    args.dataset_type = 'FSC'
    model = Model.load_from_checkpoint(ckpt_path, args=vars(args), all_classes=[], map_location=device, strict=False)
    model.eval()
    model = model.float()
    return model.to(device)


def predict(model, image_tensor, prompt, device='cuda'):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        density = model.model(image_tensor, [f'a photo of {prompt}'])
    return float(torch.sum(density / SCALE_FACTOR).item()), density.cpu()


def load_tpc268_samples(split='test', num_samples=100, seed=42):
    random.seed(seed)
    split_file = os.path.join(TPC_SPLITS, f'tpc268_{split}.txt')
    with open(split_file, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]
    samples = []
    for rel_path in all_paths:
        parts = rel_path.replace('\\', '/').split('/')
        if len(parts) >= 2:
            samples.append({'genus_species': parts[0], 'organization': parts[1], 'rel_path': rel_path, 'img_file': os.path.basename(rel_path)})
    selected = random.sample(samples, min(num_samples, len(samples)))
    with open(TPC_ANNO, 'r') as f:
        annotations = json.load(f)
    for s in selected:
        anno = annotations.get(s['img_file'], {})
        s['gt_count'] = len(anno.get('points', []))
        s['prompt'] = s['organization']
        s['path'] = os.path.join(TPC_DATA_DIR, s['rel_path'])
    return selected


def save_visualizations(pred_counts, gt_counts, samples, output_dir, results_dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    preds = np.array(pred_counts)
    gts = np.array(gt_counts)
    errors = preds - gts

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(gts, preds, alpha=0.5, s=20)
    lo, hi = min(min(gts), min(preds), 0), max(max(gts), max(preds), 1)
    plt.plot([lo, hi], [lo, hi], 'r--', lw=1)
    plt.xlabel('Ground Truth Count'); plt.ylabel('Predicted Count')
    plt.title(f'CLIP-Count on TPC-268 (n={len(preds)})\nMAE={results_dict["MAE"]:.2f}, RMSE={results_dict["RMSE"]:.2f}, R2={results_dict["R2"]:.3f}')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=150); plt.close()

    # Error histogram
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30, alpha=0.7, color='steelblue')
    plt.xlabel('Error (Pred - GT)'); plt.ylabel('Frequency')
    plt.title(f'Error Distribution (MAE={results_dict["MAE"]:.2f})')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=150); plt.close()

    # Worst and best predictions (visual)
    abs_errors = np.abs(errors)
    n_examples = min(5, len(preds))
    worst_idx = np.argsort(-abs_errors)[:n_examples]
    best_idx = np.argsort(abs_errors)[:n_examples]
    fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, 8))
    for row, idx_list in enumerate([best_idx, worst_idx]):
        for col, idx in enumerate(idx_list):
            ax = axes[row, col]
            ax.axis('off')
            s = samples[idx]
            try:
                ax.imshow(Image.open(s['path']).convert('RGB'))
            except:
                pass
            ax.set_title(f'GT={s["gt_count"]}, Pred={preds[idx]:.0f}\n{s["prompt"]}', fontsize=8)
    plt.suptitle(f'Best Predictions (top) and Worst Predictions (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_predictions.png'), dpi=150); plt.close()

    # Metrics file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write('CLIP-Count on TPC-268\n' + '='*40 + '\n')
        for k, v in results_dict.items():
            f.write(f'{k}: {v:.4f}\n')

    # CSV
    with open(os.path.join(output_dir, 'predictions.csv'), 'w') as f:
        f.write('image,species,organization,gt_count,pred_count,abs_error\n')
        for i, s in enumerate(samples):
            f.write(f'{s["img_file"]},{s["genus_species"]},{s["organization"]},{s["gt_count"]},{preds[i]:.2f},{abs_errors[i]:.2f}\n')
    print(f'Results saved to {output_dir}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    samples = load_tpc268_samples(split='test', num_samples=args.num_images, seed=args.seed)
    print(f'Loaded {len(samples)} samples')

    ckpt = os.path.join(WEIGHTS_DIR, 'clip_count_pretrained.ckpt')
    print('Loading CLIP-Count model...')
    model = load_model(ckpt, device)

    print('Running inference...')
    preds, gts = [], []
    for i, s in enumerate(samples):
        try:
            img = preprocess_image(s['path'])
            pred, _ = predict(model, img, s['prompt'], device)
            preds.append(pred); gts.append(s['gt_count'])
            if (i+1) % 10 == 0:
                print(f'  [{i+1}/{len(samples)}] {s["img_file"]}: GT={s["gt_count"]}, Pred={pred:.2f}')
        except Exception as e:
            print(f'  Error on {s["img_file"]}: {e}')
            preds.append(0); gts.append(s['gt_count'])

    results = compute_metrics(preds, gts)
    print('\n' + '='*40 + '\nRESULTS\n' + '='*40)
    for k, v in results.items():
        print(f'  {k}: {v:.4f}')
    print('\n[FLOPS] CLIP-Count: ~50 GFLOPs per 384x384 image (estimated)')

    save_visualizations(preds, gts, samples, args.output_dir, results)
    print(f'\nDone. Results in {args.output_dir}')
    return results

if __name__ == '__main__':
    main()
