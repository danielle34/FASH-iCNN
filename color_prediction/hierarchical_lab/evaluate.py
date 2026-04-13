"""
Evaluation utilities for copalette_hierarchical_lab.

  - extract_features:   run a frozen MultimodalModel over a loader, return
                        cached numpy arrays of fusion / face+clothing
                        features plus aligned label arrays
  - lab_metrics:        ΔE76, ΔE00 (mean and median), BK accuracy from
                        predicted LAB
  - results CSV I/O with skip-if-exists
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

from colors import (
    delta_e_cie76, delta_e_ciede2000, lab_to_berlin_kay,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction over a frozen multimodal trunk
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, loader, device):
    """
    Run loader through the frozen MultimodalModel.

    Returns dict with numpy / list arrays:
        fusion_features        (N, fusion_dim)        — used by Stage 2 head
        face_clothing_features (N, 2560)              — used by Stage 3 head
        bk_logits              (N, 9)                 — Stage 1 head logits
        bk_pred                (N,) int               — Stage 1 argmax
        bk_true                (N,) int               — true chromatic BK
        true_lab               (N, 3) float32         — c1 LAB ground truth
        true_css               list[str]              — c1 CSS ground truth
        image_id               list[str]
    """
    model.eval()
    use_amp = device.type == "cuda"

    fusion_chunks = []
    fc_chunks = []
    logit_chunks = []
    bk_true_chunks = []
    lab_chunks = []
    css_all = []
    ids_all = []

    for face, cloth, designer_id, bk_label, true_lab, image_id, true_css in loader:
        face = face.to(device); cloth = cloth.to(device)
        designer_id = designer_id.to(device)

        if use_amp:
            with autocast():
                fusion = model.features(face, cloth, designer_id)
                fc_feat = model.face_clothing_features(face, cloth)
                logits = model.head(fusion)
        else:
            fusion = model.features(face, cloth, designer_id)
            fc_feat = model.face_clothing_features(face, cloth)
            logits = model.head(fusion)

        fusion_chunks.append(fusion.float().cpu().numpy())
        fc_chunks.append(fc_feat.float().cpu().numpy())
        logit_chunks.append(logits.float().cpu().numpy())
        bk_true_chunks.append(bk_label.numpy())
        lab_chunks.append(true_lab.numpy())
        css_all.extend([str(c) for c in true_css])
        ids_all.extend([str(i) for i in image_id])

    fusion_features = np.concatenate(fusion_chunks, axis=0)
    fc_features = np.concatenate(fc_chunks, axis=0)
    bk_logits = np.concatenate(logit_chunks, axis=0)
    bk_pred = bk_logits.argmax(axis=1)
    bk_true = np.concatenate(bk_true_chunks, axis=0)
    true_lab = np.concatenate(lab_chunks, axis=0)

    return {
        "fusion_features": fusion_features,
        "face_clothing_features": fc_features,
        "bk_logits": bk_logits,
        "bk_pred": bk_pred,
        "bk_true": bk_true,
        "true_lab": true_lab,
        "true_css": css_all,
        "image_id": ids_all,
    }


# ---------------------------------------------------------------------------
# LAB metrics
# ---------------------------------------------------------------------------
def lab_metrics(pred_lab, true_lab):
    """Return dict with ΔE76 / ΔE00 (mean & median) + BK accuracy."""
    de76 = delta_e_cie76(pred_lab, true_lab)
    de00 = delta_e_ciede2000(pred_lab, true_lab)
    pred_bk = lab_to_berlin_kay(pred_lab)
    true_bk = lab_to_berlin_kay(true_lab)
    bk_acc = float(np.mean([p == t for p, t in zip(pred_bk, true_bk)]))
    return {
        "delta_e_cie76_mean":   float(np.mean(de76)),
        "delta_e_cie76_median": float(np.median(de76)),
        "delta_e_ciede2000_mean":   float(np.mean(de00)),
        "delta_e_ciede2000_median": float(np.median(de00)),
        "bk_accuracy_from_lab": bk_acc,
        "num_samples": int(len(pred_lab)),
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------
def save_result_row(results_csv, row_dict):
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
    else:
        df = pd.DataFrame()
    if "condition" in row_dict and "condition" in df.columns:
        mask = df["condition"] == row_dict["condition"]
        if mask.any():
            for k, v in row_dict.items():
                df.loc[mask, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(results_csv, index=False)
    logger.info(f"Saved {row_dict.get('condition', '?')} -> {results_csv}")


def load_existing_result(results_csv, condition_name):
    if not os.path.exists(results_csv):
        return None
    df = pd.read_csv(results_csv)
    if "condition" not in df.columns:
        return None
    mask = df["condition"] == condition_name
    if not mask.any():
        return None
    return df.loc[mask].iloc[0].to_dict()


def log_metrics(metrics, prefix=""):
    lines = [f"{prefix} Results:"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    logger.info("\n".join(lines))


def log_summary_table(results_csv, title):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    if len(df) == 0:
        return
    logger.info("")
    logger.info(f"---- {title} (running summary) ----")
    logger.info("\n" + df.to_string(index=False))
    logger.info("-" * len(title))
