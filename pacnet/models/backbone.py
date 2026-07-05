"""
Backbone Feature Extractor with Feature Pyramid Network (FPN)
Inspired by TasselNetV4's multi-scale cross-scene approach.
Extracts features at multiple scales to handle plant organ scale variation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetFPN(nn.Module):
    """ResNet-50 backbone with Feature Pyramid Network.
    Extracts features at 4 scales (P2, P3, P4, P5) to handle
    the 3-4 orders of magnitude scale variation in plant images.
    """
    def __init__(self, pretrained=True, output_stride=32):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)

        # Bottom-up pathway
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # C2: 1/4
        self.layer2 = resnet.layer2  # C3: 1/8
        self.layer3 = resnet.layer3  # C4: 1/16
        self.layer4 = resnet.layer4  # C5: 1/32

        # Top-down pathway (FPN)
        self.td1 = nn.Conv2d(2048, 256, 1)   # P5 ← C5
        self.td2 = nn.Conv2d(1024, 256, 1)   # P4 ← C4 + P5
        self.td3 = nn.Conv2d(512, 256, 1)    # P3 ← C3 + P4
        self.td4 = nn.Conv2d(256, 256, 1)    # P2 ← C2 + P3

        # Smoothing convolutions
        self.smooth1 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)

        # Positional context module
        self.context = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c2 = self.maxpool(x)

        c3 = self.layer1(c2)   # 1/4
        c4 = self.layer2(c3)   # 1/8
        c5 = self.layer3(c4)   # 1/16
        c6 = self.layer4(c5)   # 1/32

        # Top-down with lateral connections
        p5 = self.td1(c6)
        p5_up = F.interpolate(p5, size=c5.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.td2(c5) + p5_up

        p4_up = F.interpolate(p4, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.td3(c4) + p4_up

        p3_up = F.interpolate(p3, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.td4(c3) + p3_up

        # Smooth
        p2 = self.smooth1(p2)
        p3 = self.smooth2(p3)
        p4 = self.smooth3(p4)
        p5 = self.smooth4(p5)

        return {'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}


class ScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion module.
    Combines features from different FPN levels with learnable scale attention.
    """
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels * 2, 3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats_dict):
        p2, p3, p4, p5 = feats_dict['p2'], feats_dict['p3'], feats_dict['p4'], feats_dict['p5']
        # Upsample all to P2 resolution
        target_size = p2.shape[-2:]
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

        concat = torch.cat([p2, p3_up, p4_up, p5_up], dim=1)
        return self.fusion(concat)
