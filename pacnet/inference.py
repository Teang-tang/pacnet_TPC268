"""Single-image inference with PACNet."""
import sys, os, argparse
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pacnet"))
from models.pacnet import PACNet


def preprocess_image(image_path, size=384):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(image_path).convert("RGB")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PACNet(pretrained_backbone=False).to(device).eval()
    if args.weights and os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
    img = preprocess_image(args.image_path).to(device)
    with torch.no_grad():
        out = model(img)
        count = out["density"].sum().item()
    print(f"Predicted count: {count:.2f}")
    if args.output:
        import matplotlib.pyplot as plt
        density = out["density"][0, 0].cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(Image.open(args.image_path).convert("RGB"))
        ax1.set_title("Input")
        ax2.imshow(density, cmap="hot")
        ax2.set_title(f"Density (count={count:.1f})")
        plt.tight_layout(); plt.savefig(args.output, dpi=150)

if __name__ == "__main__":
    main()