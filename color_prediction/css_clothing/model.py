"""
Dual-stream face + clothing CSS classifier.

Mirrors copalette_crop_color.DualCropModel: each enabled stream is an
independent EfficientNet-B0 (separate weights). Disabled streams are
omitted from the fusion vector entirely so the head parameter count stays
appropriate (1280 single-stream, 2560 dual).
Self-contained — no imports from other copalette modules.
"""

import torch
import torch.nn as nn
from torchvision import models


def _make_efficientnet_b0():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    bb = models.efficientnet_b0(weights=weights)
    out_dim = bb.classifier[1].in_features  # 1280
    bb.classifier = nn.Identity()
    return bb, out_dim


class DualCropCSSModel(nn.Module):
    def __init__(self, num_classes, use_face=True, use_clothing=True):
        super().__init__()
        assert use_face or use_clothing, "At least one stream must be enabled"
        self.use_face = use_face
        self.use_clothing = use_clothing

        if use_face:
            self.face_backbone, self.face_dim = _make_efficientnet_b0()
        else:
            self.face_backbone = None
            self.face_dim = 0

        if use_clothing:
            self.clothing_backbone, self.clothing_dim = _make_efficientnet_b0()
        else:
            self.clothing_backbone = None
            self.clothing_dim = 0

        fusion_dim = self.face_dim + self.clothing_dim
        self.fusion_dim = fusion_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, face_img, clothing_img):
        parts = []
        if self.use_face and self.face_backbone is not None:
            parts.append(self.face_backbone(face_img))
        if self.use_clothing and self.clothing_backbone is not None:
            parts.append(self.clothing_backbone(clothing_img))
        x = torch.cat(parts, dim=1)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
