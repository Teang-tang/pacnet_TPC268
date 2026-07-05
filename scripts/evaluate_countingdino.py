import sys, os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'baselines' / 'countingdino'))

from src.model import VisualBackbone
from src.utils import (convert_4corners_to_x1y1x2y2, get_counting_metrics, get_features, 
    bboxes_tointeger, rescale_tensor, resize_conv_maps, ellipse_coverage, rescale_bbox)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TPC_DATA_DIR = str(PROJECT_ROOT / 'TPC-268' / 'TPC-268')
TPC_SPLITS = str(PROJECT_ROOT / 'TPC-268-main' / 'splits')
TPC_ANNO = str(PROJECT_ROOT / 'TPC-268-main' / 'annotations' / 'tpc268_annotations.json')
OUTPUT_DIR = str(PROJECT_ROOT / 'results' / 'countingdino')

class Config:
    def __init__(self):
        self.divide_et_impera = True
        self.divide_et_impera_twice = True
        self.exemplar_avg = False
        self.cosine_similarity = False
        self.normalize_features = False
        self.use_roi_norm = True
        self.roi_norm_after_mean = True
        self.use_minmax_norm = True
        self.correct_bbox_resize = True
        self.scaling_coeff = 1.0
        self.filter_background = True
        self.ellipse_normalization = True
        self.ellipse_kernel_cleaning = True

def load_samples(split='test', num_samples=100, seed=42):
    random.seed(seed)
    with open(TPC_ANNO) as f:
        annotations = json.load(f)
    split_file = os.path.join(TPC_SPLITS, 'tpc268_{}.txt'.format(split))
    with open(split_file) as f:
        split_fnames = set(os.path.basename(p.strip()) for p in f if p.strip())
    examples = {k: v for k, v in annotations.items() if k in split_fnames}
    selected = random.sample(list(examples.keys()), min(num_samples, len(examples)))
    return {k: examples[k] for k in selected}

def find_image(img_filename):
    for root, dirs, files in os.walk(TPC_DATA_DIR):
        if img_filename in files:
            return os.path.join(root, img_filename)
    return None

def process_example(img_filename, entry, model, transform, map_keys, config):
    img_path = find_image(img_filename)
    if img_path is None:
        return 0, 0, None
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    gt_count = len(entry['points'])
    with torch.no_grad():
        feats = get_features(model, img, transform, map_keys,
            divide_et_impera=config.divide_et_impera,
            divide_et_impera_twice=config.divide_et_impera_twice)
    ex_bboxes = [convert_4corners_to_x1y1x2y2(b) for b in entry['box_examples_coordinates']]
    bboxes = np.array([(x1/w, y1/h, x2/w, y2/h) for x1,y1,x2,y2 in ex_bboxes]) * feats.shape[-1]
    bboxes = bboxes_tointeger(bboxes, False)
    conv_maps, pooled_features_list, output_sizes, rescaled_bboxes = [], [], [], []
    for bbox in bboxes:
        bbox_t = torch.tensor(bbox)
        osize = (int(bbox_t[3]-bbox_t[1]), int(bbox_t[2]-bbox_t[0]))
        pooled = ops.roi_align(feats, [bbox_t.unsqueeze(0).float().to(device)], output_size=osize, spatial_scale=1.0)
        if config.ellipse_kernel_cleaning:
            ell = ellipse_coverage(pooled.shape[-2],pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(device)
            pooled *= ell
        pooled_features_list.append(pooled)
        conv_weights = pooled.view(feats.shape[1], 1, *osize)
        conv_layer = nn.Conv2d(feats.shape[1], feats.shape[1], kernel_size=osize, padding=0,
            groups=feats.shape[1], bias=False).to(device)
        conv_layer.weight = nn.Parameter(pooled.view(feats.shape[1], 1, *osize))
        with torch.no_grad():
            output = conv_layer(feats[0])
        rescaled_bbox = rescale_bbox(bbox_t, output, feats) if config.correct_bbox_resize else bbox_t
        rescaled_bboxes.append(rescaled_bbox)
        conv_maps.append(output)
        output_sizes.append(osize)
    output = post_process(conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config)
    density_map = None
    try:
        density_map = output.cpu().numpy()
    except:
        pass
    return gt_count, output.sum().item(), density_map

def post_process(conv_maps, pooled_feats, bboxes, output_sizes, config):
    output, ratios = resize_conv_maps(conv_maps)
    output = output.mean(dim=0)
    if config.use_minmax_norm:
        output = rescale_tensor(output)
    pooled_vals = []
    for bbox, ratio in zip(bboxes, ratios):
        scaled = torch.tensor([bbox[0]*ratio[1], bbox[1]*ratio[0], bbox[2]*ratio[1], bbox[3]*ratio[0]]).int()
        osize = (int(scaled[3]-scaled[1]), int(scaled[2]-scaled[0]))
        p = ops.roi_align(output.unsqueeze(0).unsqueeze(0),
            [scaled.unsqueeze(0).float().to(device)], output_size=osize, spatial_scale=1.0)
        pooled_vals.append(p)
    if config.ellipse_normalization:
        norm_coeff = sum([(p[0,0]*ellipse_coverage(p.shape[-2],p.shape[-1]).to(device)).sum() for p in pooled_vals]) / (len(pooled_vals)*config.scaling_coeff)
    else:
        norm_coeff = sum([p.sum() for p in pooled_vals]) / (len(pooled_vals)*config.scaling_coeff)
    output = output / norm_coeff
    if config.filter_background:
        thresh = max(f.shape[-2]*f.shape[-1] for f in pooled_feats)
        output[output < (1/thresh)*1.0] = 0
    return output

def estimate_flops(model, img_size=840, patch_size=14):
    """Estimate FLOPS for CountingDINO with ViT-B/14"""
    n_patches = (img_size // patch_size) ** 2
    hidden_dim = 768
    n_layers = 12
    n_heads = 12
    mlp_dim = hidden_dim * 4
    
    # Self-attention: QKV + attention + projection
    attn_flops = 2 * n_layers * n_patches * (3 * hidden_dim * hidden_dim + n_patches * hidden_dim + hidden_dim * hidden_dim)
    # MLP: two linear layers
    mlp_flops = 2 * n_layers * n_patches * (hidden_dim * mlp_dim + mlp_dim * hidden_dim)
    # Patch embedding
    patch_flops = n_patches * 3 * (patch_size ** 2) * hidden_dim
    
    total_gflops = (attn_flops + mlp_flops + patch_flops) / 1e9
    
    # divide_et_impera processes 4 overlapping regions (rough estimate)
    total_gflops *= 2.5
    
    return total_gflops

def save_results(preds, gts, samples, output_dir, results_dict, flops, error_cases=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    preds_a = np.array(preds)
    gts_a = np.array(gts)
    errors = preds_a - gts_a
    abs_errors = np.abs(errors)
    mean_gt = np.mean(gts_a)
    
    # Compute all metrics
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((gts_a - np.mean(gts_a)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    rmae = mae / (mean_gt + 1e-10)
    rrmse = rmse / (mean_gt + 1e-10)
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(gts_a, preds_a, alpha=0.5, s=20, c='steelblue')
    lo, hi = 0, max(max(gts_a), max(preds_a), 1) * 1.05
    plt.plot([lo, hi], [lo, hi], 'r--', lw=1.5, alpha=0.7)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel('Ground Truth', fontsize=12)
    plt.ylabel('Predicted Count', fontsize=12)
    plt.title('CountingDINO on TPC-268 (n={})\nMAE={:.2f}, RMSE={:.2f}, R2={:.3f}'.format(len(preds), mae, rmse, r2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=150)
    plt.close()
    
    # Error histogram
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
    plt.xlabel('Error (Pred - GT)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution (MAE={:.2f}, rMAE={:.3f})'.format(mae, rmae))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=150)
    plt.close()
    
    # Metrics file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write('CountingDINO on TPC-268 ({} images)\n'.format(len(preds)))
        f.write('='*50 + '\n')
        f.write('MAE:  {:.4f}\n'.format(mae))
        f.write('RMSE: {:.4f}\n'.format(rmse))
        f.write('R2:   {:.4f}\n'.format(r2))
        f.write('rMAE: {:.4f}\n'.format(rmae))
        f.write('rRMSE:{:.4f}\n'.format(rrmse))
        f.write('FLOPS:{:.2f} GFLOPs (estimated)\n'.format(flops))
        f.write('Mean_GT:  {:.4f}\n'.format(mean_gt))
        f.write('Mean_Pred:{:.4f}\n'.format(float(np.mean(preds_a))))
    
    # CSV
    with open(os.path.join(output_dir, 'predictions.csv'), 'w') as f:
        f.write('image,gt_count,pred_count,abs_error\n')
        for i, k in enumerate(samples):
            f.write('{},{},{:.2f},{:.2f}\n'.format(k, gts[i], preds[i], abs_errors[i]))
    
    # Error case visualization (worst 5)
    if error_cases:
        n_cases = min(5, len(error_cases))
        fig, axes = plt.subplots(n_cases, 2, figsize=(12, 4*n_cases))
        if n_cases == 1:
            axes = [axes]
        for i, (img_name, gt, pred, dm) in enumerate(error_cases[:n_cases]):
            img_path = find_image(img_name)
            if img_path:
                img = Image.open(img_path).convert('RGB')
                ax_img = axes[i][0] if n_cases > 1 else axes[i]
                ax_img.imshow(img)
                ax_img.set_title('{}: GT={}, Pred={:.0f}'.format(img_name, gt, pred), fontsize=10)
                ax_img.axis('off')
            if dm is not None and dm.size > 0:
                ax_dm = axes[i][1] if n_cases > 1 else axes[i+1]
                try:
                    dm_resized = np.maximum(dm.squeeze(), 0)
                    ax_dm.imshow(dm_resized, cmap='hot', interpolation='nearest')
                    ax_dm.set_title('Density Map (sum={:.0f})'.format(pred), fontsize=10)
                    ax_dm.axis('off')
                except:
                    ax_dm.text(0.5, 0.5, 'No density map', ha='center')
                    ax_dm.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_case_examples.png'), dpi=150)
        plt.close()
    
    print('Results saved to {}'.format(output_dir))
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'rMAE': rmae, 'rRMSE': rrmse, 'FLOPS': flops}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14_reg')
    parser.add_argument('--num_images', type=int, default=700)
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print('Device: {}'.format(device))
    print('Loading {}...'.format(args.model_name))
    config = Config()
    resize_dim = 840 if 'dinov2' in args.model_name else 480
    
    model = VisualBackbone(args.model_name, img_size=resize_dim).to(device).eval()
    transform = T.Compose([
        T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    map_keys = ['vit_out']
    
    # Load samples
    samples = load_samples('test', args.num_images, args.seed)
    print('Loaded {} samples'.format(len(samples)))
    
    # Estimate FLOPS
    flops = estimate_flops(model, resize_dim)
    print('Estimated FLOPS: {:.2f} GFLOPs per image'.format(flops))
    
    # Run inference
    print('\nRunning inference...')
    preds, gts = [], []
    error_cases = []
    
    for idx, (fname, entry) in enumerate(tqdm(samples.items(), dynamic_ncols=True)):
        gt, pred, dm = process_example(fname, entry, model, transform, map_keys, config)
        preds.append(pred)
        gts.append(gt)
        if (idx + 1) % 100 == 0:
            print('  [{}/{}] {}: GT={}, Pred={:.2f}'.format(idx+1, len(samples), fname, gt, pred))
    
    # Find worst error cases
    abs_errors = [abs(preds[i] - gts[i]) for i in range(len(preds))]
    worst_idx = np.argsort(abs_errors)[-10:][::-1]
    error_cases = []
    for idx in worst_idx:
        fname = list(samples.keys())[idx]
        gt, pred, _ = process_example(fname, samples[fname], model, transform, map_keys, config)
        error_cases.append((fname, gts[idx], preds[idx], None))
    
    # Compute and save results
    all_metrics = save_results(preds, gts, list(samples.keys()), args.output_dir, {}, flops, error_cases)
    
    print('\n' + '='*50)
    print('FINAL RESULTS')
    print('='*50)
    print('  MAE:   {:.4f}'.format(all_metrics['MAE']))
    print('  RMSE:  {:.4f}'.format(all_metrics['RMSE']))
    print('  R2:    {:.4f}'.format(all_metrics['R2']))
    print('  rMAE:  {:.4f}'.format(all_metrics['rMAE']))
    print('  rRMSE: {:.4f}'.format(all_metrics['rRMSE']))
    print('  FLOPS: {:.2f} GFLOPs'.format(all_metrics['FLOPS']))
    print('\nDone. Results in {}'.format(args.output_dir))

if __name__ == '__main__':
    main()
