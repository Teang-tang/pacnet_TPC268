"""PACNet 评估脚本（CPU）"""
import sys, os, warnings, json, random
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms as T
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pacnet"))
from models.pacnet import PACNet
from utils.metrics import compute_metrics
DATA = str(ROOT / "TPC-268" / "TPC-268")
SPLIT = str(ROOT / "TPC-268-main" / "splits" / "tpc268_test.txt")
ANNO = str(ROOT / "TPC-268-main" / "annotations" / "tpc268_annotations.json")
WEIGHT = str(ROOT / "results" / "pacnet" / "best_model.pth")
OUT = str(ROOT / "results" / "pacnet")
N_IMAGES = 200
def main():
    with open(ANNO) as f: annotations = json.load(f)
    with open(SPLIT) as f: paths = [l.strip() for l in f if l.strip()]
    random.seed(42); samples = random.sample(paths, min(N_IMAGES, len(paths)))
    device = "cpu"
    model = PACNet(pretrained_backbone=True)
    ckpt = torch.load(WEIGHT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    epoch = ckpt.get("epoch", "?")
    print(f"Model loaded from epoch {epoch}")
    transform = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    preds, gts = [], []
    for rel_path in tqdm(samples):
        parts = rel_path.replace("\\", "/").split("/")
        img_file = os.path.basename(rel_path)
        img_path = os.path.join(DATA, rel_path)
        anno = annotations.get(img_file, {}); pts = anno.get("points", []); gt = len(pts)
        if gt == 0: continue
        try:
            img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad(): out = model(img)
            preds.append(float(out["density"].sum(dim=(2,3)).item())); gts.append(gt)
        except Exception as e:
            print(f"  Error {img_file}: {e}")
    results = compute_metrics(preds, gts)
    print("\n" + "="*50 + "\nPACNet EVALUATION RESULTS\n" + "="*50)
    for k, v in results.items(): print(f"  {k}: {v:.4f}")
    print(f"  Samples: {len(preds)}")
    with open(os.path.join(OUT, "metrics.txt"), "w") as f:
        f.write(f"PACNet on TPC-268 Test\nTrained on 500 images, 15 epochs, CPU\n")
        for k, v in results.items(): f.write(f"{k}: {v:.4f}\n")
        f.write(f"N: {len(preds)}\n")
    print(f"Saved to {OUT}")
if __name__ == "__main__": main()
