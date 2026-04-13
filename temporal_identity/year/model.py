"""
EfficientNet-B0 year/decade classifier.

Single-stream model: clothing crop -> 1280 -> 512 -> num_classes.
Same architecture for both the 34-class year head and the 4-class decade head.
Self-contained — no imports from other copalette modules.
"""

import torch.nn as nn
from torchvision import models


class YearClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        bb = models.efficientnet_b0(weights=weights)
        self.backbone_dim = bb.classifier[1].in_features  # 1280
        bb.classifier = nn.Identity()
        self.backbone = bb

        self.head = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, img):
        return self.head(self.backbone(img))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
