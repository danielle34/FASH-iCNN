"""
Dual-stream face + silhouette model for designer classification.

Each enabled stream is an independent EfficientNet-B0 (separate weights).
Disabled streams are omitted from the fusion vector entirely so the head
parameter count stays appropriate (1280 single-stream, 2560 dual).
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


class FaceSilhouetteModel(nn.Module):
    """Face + silhouette dual-stream classifier with optional ablation."""

    def __init__(self, num_classes, use_face=True, use_silhouette=True):
        super().__init__()
        assert use_face or use_silhouette, "At least one stream must be enabled"
        self.use_face = use_face
        self.use_silhouette = use_silhouette

        if use_face:
            self.face_backbone, self.face_dim = _make_efficientnet_b0()
        else:
            self.face_backbone = None
            self.face_dim = 0

        if use_silhouette:
            self.silhouette_backbone, self.silhouette_dim = _make_efficientnet_b0()
        else:
            self.silhouette_backbone = None
            self.silhouette_dim = 0

        fusion_dim = self.face_dim + self.silhouette_dim
        self.fusion_dim = fusion_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, face_img, silhouette_img):
        parts = []
        if self.use_face and self.face_backbone is not None:
            parts.append(self.face_backbone(face_img))
        if self.use_silhouette and self.silhouette_backbone is not None:
            parts.append(self.silhouette_backbone(silhouette_img))
        x = torch.cat(parts, dim=1)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
