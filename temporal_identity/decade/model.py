"""
EfficientNet-B0 model for copalette_clothing_decade.
"""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DecadeModel(nn.Module):
    """EfficientNet-B0 -> 1280 -> 256 -> num_classes."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.backbone_dim = 1280
        self.head = nn.Sequential(
            nn.Linear(self.backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img):
        return self.head(self.backbone(img))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
