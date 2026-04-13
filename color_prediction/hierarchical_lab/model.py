"""
Models for copalette_hierarchical_lab.

Three model classes:

  - MultimodalModel: replicates the upstream copalette_multimodal_color
    architecture exactly so the F checkpoint loads cleanly via
    `load_state_dict`. Used as Stage 1 (BK 9-class predictor) and as the
    frozen feature extractor for Stages 2 and 3.

  - FamilyCSSClassifier: a small head over the frozen multimodal features
    that classifies a CSS color within a single Berlin-Kay family. One
    instance per family.

  - LABRegressor: regression head over the frozen face+clothing 2560-dim
    feature vector that predicts a constrained LAB residual:
        predicted_LAB = css_centroid + tanh(head(features)) * max_offset
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


# ---------------------------------------------------------------------------
# Multimodal model — must mirror copalette_multimodal_color exactly
# ---------------------------------------------------------------------------
class MultimodalModel(nn.Module):
    """
    Face + clothing + designer fusion classifier. Mirrors
    copalette_multimodal_color/model.py:MultimodalModel so the F checkpoint
    loads cleanly. Set num_classes=9 (chromatic BK) for the F head.
    """

    def __init__(self, num_classes, num_designers,
                 use_face=True, use_clothing=True, use_designer=True):
        super().__init__()
        assert use_face or use_clothing or use_designer
        self.use_face = use_face
        self.use_clothing = use_clothing
        self.use_designer = use_designer

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

        if use_designer:
            self.designer_dim = 32
            self.designer_embed = nn.Embedding(num_designers, self.designer_dim)
        else:
            self.designer_dim = 0
            self.designer_embed = None

        fusion_dim = self.face_dim + self.clothing_dim + self.designer_dim
        self.fusion_dim = fusion_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, face_img, clothing_img, designer_id):
        parts = []
        if self.use_face and self.face_backbone is not None:
            parts.append(self.face_backbone(face_img))
        if self.use_clothing and self.clothing_backbone is not None:
            parts.append(self.clothing_backbone(clothing_img))
        if self.use_designer and self.designer_embed is not None:
            parts.append(self.designer_embed(designer_id))
        x = torch.cat(parts, dim=1)
        return self.head(x)

    @torch.no_grad()
    def features(self, face_img, clothing_img, designer_id):
        """Return the pre-head fusion vector (used for Stages 2 and 3)."""
        parts = []
        if self.use_face and self.face_backbone is not None:
            parts.append(self.face_backbone(face_img))
        if self.use_clothing and self.clothing_backbone is not None:
            parts.append(self.clothing_backbone(clothing_img))
        if self.use_designer and self.designer_embed is not None:
            parts.append(self.designer_embed(designer_id))
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def face_clothing_features(self, face_img, clothing_img):
        """Return the 2560-dim face+clothing concat (no designer). Used by Stage 3."""
        face_feat = self.face_backbone(face_img)
        cloth_feat = self.clothing_backbone(clothing_img)
        return torch.cat([face_feat, cloth_feat], dim=1)


# ---------------------------------------------------------------------------
# Per-family CSS classifier (Stage 2)
# ---------------------------------------------------------------------------
class FamilyCSSClassifier(nn.Module):
    """
    Small classification head over the frozen multimodal fusion vector.
    One instance per Berlin-Kay family — output dim = number of CSS colors
    that map to that family AND are present in the training set.
    """

    def __init__(self, fusion_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, features):
        return self.head(features)


# ---------------------------------------------------------------------------
# Constrained LAB regressor (Stage 3)
# ---------------------------------------------------------------------------
class LABRegressor(nn.Module):
    """
    Predicts a constrained residual LAB offset on top of a per-image
    CSS centroid:

        offset = tanh(head(features)) * max_offset    # (B, 3) in [-max, +max]
        predicted_LAB = css_centroid + offset

    The CSS centroid is supplied externally (it depends on which CSS color
    Stage 2 picked, or the true CSS color in oracle mode).

    Input feature dim defaults to 2560 (frozen face + clothing concat).
    """

    def __init__(self, in_dim=2560, max_offset=(10.0, 15.0, 15.0)):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3),
        )
        self.register_buffer(
            "max_offset",
            torch.tensor(list(max_offset), dtype=torch.float32),
        )

    def forward(self, features, css_centroid):
        offset = torch.tanh(self.head(features)) * self.max_offset  # (B, 3)
        return css_centroid + offset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
