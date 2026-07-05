"""TPC-268 数据集加载器（带密度图生成）"""
import os, json, random, sys
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.density_map import generate_density_map_fast
class PlantAugmentation:
    def __init__(self, image_size=256):
        t = [transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.val_transform = transforms.Compose(t)
        self.train_transform = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(0.2,0.2,0.2,0.1), *t[1:]])
    def __call__(self, img, is_train=True): return self.train_transform(img) if is_train else self.val_transform(img)
class TPC268Dataset(Dataset):
    def __init__(self, data_dir, split_file, anno_file, transform=None, num_samples=None, seed=42, density_size=(96,96)):
        self.data_dir = data_dir; self.transform = transform; self.dh, self.dw = density_size
        random.seed(seed)
        with open(anno_file,'r') as f: self.annotations = json.load(f)
        with open(split_file,'r') as f: all_paths = [l.strip() for l in f if l.strip()]
        self.samples = []
        for rp in all_paths:
            parts = rp.replace('\\','/').split('/')
            if len(parts) >= 2: self.samples.append({'rel_path':rp,'species':parts[0],'organization':parts[1],'img_file':os.path.basename(rp)})
        if num_samples and 0 < num_samples < len(self.samples): self.samples = random.sample(self.samples, num_samples)
        filtered = []
        for s in self.samples:
            anno = self.annotations.get(s['img_file'], {})
            pts = anno.get('points', [])
            if len(pts) > 0:
                s['gt_count'] = len(pts); s['points'] = pts
                s['path'] = os.path.join(data_dir, s['rel_path'])
                filtered.append(s)
        self.samples = filtered
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        ow, oh = img.size
        if self.transform: img = self.transform(img, is_train=False)
        density = generate_density_map_fast(s['points'], oh, ow, self.dh, self.dw)
        return {'image':img, 'gt_count':torch.tensor(s['gt_count'],dtype=torch.float32), 'density':torch.from_numpy(density).unsqueeze(0), 'img_file':s['img_file'], 'organization':s['organization'], 'species':s['species']}
    @staticmethod
    def collate_fn(batch):
        return {'image':torch.stack([b['image'] for b in batch]), 'gt_count':torch.stack([b['gt_count'] for b in batch]).unsqueeze(1), 'density':torch.stack([b['density'] for b in batch]), 'img_file':[b['img_file'] for b in batch], 'organization':[b['organization'] for b in batch], 'species':[b['species'] for b in batch]}
