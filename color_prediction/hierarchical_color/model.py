"""
Models for hierarchical color experiments.

MultimodalBKModel — replicates the architecture from copalette_multimodal_color
so we can load ckpt_F_full.pth (face + clothing + designer -> 9 BK classes).

FamilyCSSModel — dual-stream face + clothing classifier for within-family CSS
prediction. Backbone weights are initialized from external pretrained
checkpoints (face-only and clothing-only models) and fine-tuned at lower LR.
Self-contained — no imports from other copalette modules.
"""

import logging
import os

import torch
import torch.nn as nn
from torchvision import models

log = logging.getLogger("hierarchical")


def _make_efficientnet_b0():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    bb = models.efficientnet_b0(weights=weights)
    out_dim = bb.classifier[1].in_features  # 1280
    bb.classifier = nn.Identity()
    return bb, out_dim


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: MultimodalBKModel
# ═══════════════════════════════════════════════════════════════════════════════
# Mirrors copalette_multimodal_color.MultimodalModel with all three streams on
# (use_face=True, use_clothing=True, use_designer=True) so the F_full
# checkpoint loads cleanly. Module names must match exactly.

class MultimodalBKModel(nn.Module):
    def __init__(self, num_classes, num_designers):
        super().__init__()
        self.use_face = True
        self.use_clothing = True
        self.use_designer = True

        self.face_backbone, self.face_dim = _make_efficientnet_b0()
        self.clothing_backbone, self.clothing_dim = _make_efficientnet_b0()
        self.designer_dim = 32
        self.designer_embed = nn.Embedding(num_designers, self.designer_dim)

        fusion_dim = self.face_dim + self.clothing_dim + self.designer_dim
        self.fusion_dim = fusion_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, face_img, clothing_img, designer_id):
        vis_face = self.face_backbone(face_img)
        vis_cloth = self.clothing_backbone(clothing_img)
        des = self.designer_embed(designer_id)
        x = torch.cat([vis_face, vis_cloth, des], dim=1)
        return self.head(x)


def load_stage1_checkpoint(model, checkpoint_path, device, log):
    """Load the F_full multimodal checkpoint into a MultimodalBKModel.

    Tries strict load; if it fails, falls back to non-strict and logs the
    mismatched keys so the user can see what didn't transfer.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
        log.info(f"  Loaded Stage 1 checkpoint (strict) from {checkpoint_path}")
    except RuntimeError as e:
        log.warning(f"  Strict load failed: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.warning(f"  Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if len(missing) <= 20:
            log.warning(f"  Missing: {missing}")
        if len(unexpected) <= 20:
            log.warning(f"  Unexpected: {unexpected}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: FamilyCSSModel
# ═══════════════════════════════════════════════════════════════════════════════

class FamilyCSSModel(nn.Module):
    """Dual-stream face + clothing classifier for within-family CSS prediction."""

    def __init__(self, num_classes):
        super().__init__()
        self.face_backbone, self.face_dim = _make_efficientnet_b0()
        self.clothing_backbone, self.clothing_dim = _make_efficientnet_b0()

        fusion_dim = self.face_dim + self.clothing_dim  # 2560
        self.fusion_dim = fusion_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, face_img, clothing_img):
        vis_face = self.face_backbone(face_img)
        vis_cloth = self.clothing_backbone(clothing_img)
        x = torch.cat([vis_face, vis_cloth], dim=1)
        return self.head(x)


def _extract_backbone_state(state, prefix):
    """Extract a state dict subset with the given prefix and strip the prefix.

    e.g. with prefix='face_backbone.', returns {k[len(prefix):]: v for k starting with prefix}.
    """
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def init_family_backbones(model, face_ckpt_path, clothing_ckpt_path, device, log):
    """Initialize face_backbone and clothing_backbone from external checkpoints.

    The face checkpoint may come from a single-stream model (backbone is named
    'backbone.*') or a dual-stream model (named 'face_backbone.*'). We try both.
    Same for clothing.
    """
    # Face backbone
    if face_ckpt_path and os.path.exists(face_ckpt_path):
        try:
            ck = torch.load(face_ckpt_path, map_location=device, weights_only=False)
            state = ck["model_state_dict"] if "model_state_dict" in ck else ck
            face_state = _extract_backbone_state(state, "face_backbone.")
            if not face_state:
                face_state = _extract_backbone_state(state, "backbone.")
            if face_state:
                missing, unexpected = model.face_backbone.load_state_dict(
                    face_state, strict=False,
                )
                log.info(f"  Initialized face_backbone from {face_ckpt_path} "
                         f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                log.warning(f"  No face backbone keys found in {face_ckpt_path}; "
                            f"keeping ImageNet init")
        except Exception as e:
            log.warning(f"  Failed to load face checkpoint {face_ckpt_path}: {e}")
    else:
        log.warning(f"  Face checkpoint not found: {face_ckpt_path}")

    # Clothing backbone
    if clothing_ckpt_path and os.path.exists(clothing_ckpt_path):
        try:
            ck = torch.load(clothing_ckpt_path, map_location=device, weights_only=False)
            state = ck["model_state_dict"] if "model_state_dict" in ck else ck
            cloth_state = _extract_backbone_state(state, "clothing_backbone.")
            if not cloth_state:
                cloth_state = _extract_backbone_state(state, "backbone.")
            if cloth_state:
                missing, unexpected = model.clothing_backbone.load_state_dict(
                    cloth_state, strict=False,
                )
                log.info(f"  Initialized clothing_backbone from {clothing_ckpt_path} "
                         f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                log.warning(f"  No clothing backbone keys found in {clothing_ckpt_path}; "
                            f"keeping ImageNet init")
        except Exception as e:
            log.warning(f"  Failed to load clothing checkpoint {clothing_ckpt_path}: {e}")
    else:
        log.warning(f"  Clothing checkpoint not found: {clothing_ckpt_path}")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
