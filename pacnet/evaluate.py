"""
PACNet Evaluation Script for TPC-268 Dataset.
Evaluates the Plant-Adaptive Counting Network and produces metrics
comparable with the three baselines (CLIP-Count, CountingDINO, CountGD).

Usage:
    python pacnet/evaluate.py [--num_images 100] [--output_dir results/pacnet]

References:
    [2] TasselNetV4: Cross-scene, cross-scale, cross-species plant counting
    [3] TasselNet: Local counts regression network for plant counting
"""
import sys, os, json, random, argparse, time
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'pacnet'))

from models.pacnet import PACNet
from models.backbone import ResNetFPN, ScaleFeatureFusion
from models.multi_scale_head import ScaleAdaptiveHeads
from models.local_refine import LocalRefineModule
from datasets.plant_dataset import TPC268Dataset, PlantAugmentation
from utils.metrics import compute_metrics, print_results, estimate_flops
from utils.visualization import (
    save_scatter_plot, save_error_histogram, save_example_predictions,
    save_scale_attention_visualization
)
from utils.config import (
    TPC_DATA_DIR, TPC_SPLITS, TPC_ANNO, OUTPUT_DIR, WEIGHTS_DIR, IMAGE_SIZE
)


def load_model(device='cpu', pretrained_weights=None):
    """Load PACNet model with optional pretrained weights."""
    model = PACNet(pretrained_backbone=True)

    if pretrained_weights and os.path.exists(pretrained_weights):
        checkpoint = torch.load(pretrained_weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f'Loaded weights from {pretrained_weights}')

    model.eval()
    return model.to(device)


@torch.no_grad()
def predict(model, image_tensor, device='cpu'):
    """Run PACNet inference on a single image."""
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Run model
    out = model(image_tensor, return_aux=False)
    density = out['density']

    # Sum density map to get count
    count = density.sum(dim=(2, 3)).item()

    return count, density.cpu()


def multi_scale_inference(model, image_tensor, scales=[0.75, 1.0, 1.25], device='cpu'):
    """Multi-scale test-time inference for improved accuracy."""
    counts = []
    for scale in scales:
        if scale != 1.0:
            h, w = image_tensor.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = torch.nn.functional.interpolate(
                image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            scaled = image_tensor

        pred, _ = predict(model, scaled, device)
        # Scale the count back
        if scale != 1.0:
            pred = pred / (scale ** 2)
        counts.append(pred)

    return np.mean(counts), np.std(counts)


def main():
    parser = argparse.ArgumentParser(description='PACNet Evaluation on TPC-268')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of test images (-1 for all)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pretrained weights (optional)')
    parser.add_argument('--multi_scale', action='store_true',
                        help='Use multi-scale test-time augmentation')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    print('Loading TPC-268 dataset...')
    transform = PlantAugmentation(image_size=IMAGE_SIZE)
    dataset = TPC268Dataset(
        data_dir=TPC_DATA_DIR,
        split_file=os.path.join(TPC_SPLITS, f'tpc268_{args.split}.txt'),
        anno_file=TPC_ANNO,
        transform=transform,
        num_samples=args.num_images if args.num_images > 0 else None,
        seed=args.seed,
    )
    print(f'Loaded {len(dataset)} samples')

    # Load model
    print('Initializing PACNet model...')
    model = load_model(device, args.pretrained_weights)

    # Estimate FLOPS
    try:
        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        flops = estimate_flops(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
    except:
        flops = estimate_flops(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
    print(f'Estimated FLOPS: {flops:.2f} GFLOPs')

    # Run inference
    print(f'\nRunning inference on {len(dataset)} images...')
    pred_counts = []
    gt_counts = []
    all_samples = []

    start_time = time.time()
    for i in tqdm(range(len(dataset)), dynamic_ncols=True):
        sample = dataset[i]
        img_tensor = sample['image'].unsqueeze(0).to(device)
        gt = sample['gt_count'].item()

        try:
            if args.multi_scale:
                pred, _ = multi_scale_inference(model, img_tensor, device=device)
            else:
                pred, _ = predict(model, img_tensor, device)

            pred_counts.append(pred)
            gt_counts.append(gt)

            all_samples.append({
                'img_file': sample['img_file'],
                'species': sample['species'],
                'organization': sample['organization'],
                'path': os.path.join(TPC_DATA_DIR, dataset.samples[i]['rel_path']),
            })

        except Exception as e:
            print(f'  Error on {sample["img_file"]}: {e}')
            pred_counts.append(0)
            gt_counts.append(gt)

    elapsed = time.time() - start_time
    print(f'Inference completed in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/image)')

    # Compute metrics
    results = compute_metrics(pred_counts, gt_counts)
    results['FLOPS'] = flops
    results['Time_Per_Image'] = elapsed / len(dataset)
    results['Model'] = 'PACNet'

    # Print results
    print_results(results, f'PACNet ({"Multi-Scale" if args.multi_scale else "Single-Scale"})', flops)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    preds_arr = np.array(pred_counts)
    gts_arr = np.array(gt_counts)
    errors = preds_arr - gts_arr
    abs_errors = np.abs(errors)

    # Metrics file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'PACNet on TPC-268 ({args.split} split, {len(pred_counts)} images)\n')
        f.write(f'Backbone: ResNet-50 + FPN + Scale-Adaptive Heads + Local Refinement\n')
        f.write(f'Model: Plant-Adaptive Counting Network (TasselNetV4+TasselNet inspired)\n')
        f.write('=' * 50 + '\n')
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f'{k}: {v:.4f}\n')
            else:
                f.write(f'{k}: {v}\n')

    # CSV predictions
    with open(os.path.join(args.output_dir, 'predictions.csv'), 'w') as f:
        f.write('image,species,organization,gt_count,pred_count,abs_error\n')
        for i, s in enumerate(all_samples):
            f.write(f'{s["img_file"]},{s["species"]},{s["organization"]},{gt_counts[i]},{preds_arr[i]:.2f},{abs_errors[i]:.2f}\n')

    # Visualizations
    save_scatter_plot(
        gts_arr, preds_arr,
        os.path.join(args.output_dir, 'scatter_plot.png'),
        title=f'PACNet on TPC-268 (n={len(preds_arr)})\nMAE={results["MAE"]:.2f}, RMSE={results["RMSE"]:.2f}, R2={results["R2"]:.3f}'
    )
    save_error_histogram(errors, os.path.join(args.output_dir, 'error_histogram.png'), results['MAE'])
    save_example_predictions(all_samples, preds_arr, gts_arr, os.path.join(args.output_dir, 'example_predictions.png'))

    # Cross-baseline comparison table
    print('\n' + '=' * 50)
    print('CROSS-BASELINE COMPARISON')
    print('=' * 50)
    print(f'{"Metric":<15} {"CLIP-Count":<15} {"CountingDINO":<15} {"CountGD":<15} {"PACNet(ours)":<15}')
    print('-' * 75)

    baseline_data = {
        'MAE': (26.84, 36.81, 26.10, results['MAE']),
        'RMSE': (52.54, 86.90, 59.19, results['RMSE']),
        'R2': (0.59, -0.30, 0.40, results['R2']),
        'rMAE': (0.39, 0.56, 0.40, results['rMAE']),
        'rRMSE': (0.76, 1.32, 0.90, results['rRMSE']),
    }

    for metric, (v1, v2, v3, v4) in baseline_data.items():
        print(f'{metric:<15} {v1:<15.4f} {v2:<15.4f} {v3:<15.4f} {v4:<15.4f}')

    print(f'\nNote: PACNet uses ResNet-50 backbone + FPN + Scale-Adaptive Heads.')
    print(f'Pretrained weights not available yet - results are from randomly initialized model.')
    print(f'After training on TPC-268 training set, metrics should improve significantly.')
    print(f'\nDone. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
