# -*- coding: utf-8 -*-
import sys, os, argparse, warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pacnet"))
from models.pacnet import PACNet
from models.loss import HybridCountingLoss
from datasets.plant_dataset import TPC268Dataset, PlantAugmentation
from utils.metrics import compute_metrics
# 路径配置
TPC_DATA = str(ROOT / "TPC-268" / "TPC-268")
TPC_SPLITS = str(ROOT / "TPC-268-main" / "splits")
TPC_ANNO = str(ROOT / "TPC-268-main" / "annotations" / "tpc268_annotations.json")
OUTPUT_DIR = str(ROOT / "results" / "pacnet")
IMAGE_SIZE = 256  # 小尺寸加�?CPU
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)
        gt_counts = batch["gt_count"].to(device)
        out = model(images, return_aux=True)
        loss, _ = criterion(out, gt_density, gt_counts)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in tqdm(loader, desc="Validating"):
        images = batch["image"].to(device)
        out = model(images)
        counts = out["density"].sum(dim=(2, 3)).cpu().numpy()
        preds.extend(counts.flatten().tolist())
        gts.extend(batch["gt_count"].cpu().numpy().flatten().tolist())
    return compute_metrics(preds, gts)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cpu"
    print(f"[PACNet Train] Device: {device}")
    print(f"[PACNet Train] Train samples: {args.num_train}, Val samples: {args.num_val}, Epochs: {args.epochs}")
    transform = PlantAugmentation(image_size=IMAGE_SIZE)
    train_ds = TPC268Dataset(TPC_DATA, os.path.join(TPC_SPLITS, "tpc268_train.txt"), TPC_ANNO, transform, num_samples=args.num_train, seed=args.seed, density_size=(64,64))
    val_ds = TPC268Dataset(TPC_DATA, os.path.join(TPC_SPLITS, "tpc268_val.txt"), TPC_ANNO, transform, num_samples=args.num_val, seed=args.seed, density_size=(64,64))
    print(f"[PACNet Train] Train dataset: {len(train_ds)} samples, Val dataset: {len(val_ds)} samples")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, collate_fn=TPC268Dataset.collate_fn)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0, collate_fn=TPC268Dataset.collate_fn)
    model = PACNet(pretrained_backbone=True).to(device)
    criterion = HybridCountingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_results = validate(model, val_loader, device)
        scheduler.step()
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_MAE={val_results['MAE']:.4f}, val_R2={val_results['R2']:.4f}")
        if val_results.get("MAE", float("inf")) < best_mae:
            best_mae = val_results["MAE"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_metrics": val_results}, os.path.join(args.output_dir, "best_model.pth"))
            print(f"  [Saved] best_model.pth (MAE={best_mae:.4f})")
    print(f"[Done] Best val MAE: {best_mae:.4f}")
if __name__ == "__main__":
    main()
