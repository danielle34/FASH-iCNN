"""
Evaluation utilities for CSS clothing experiments.

Top-k accuracy, macro F1, baseline, per-CSS-color top-1 breakdown, and the
**ΔE00 perceptual metric** between predicted CSS centroid and true CSS
centroid in LAB space (the headline metric for this folder).
Self-contained — no imports from other copalette modules.
"""

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from colors import (
    delta_e_ciede2000, delta_e_cie76,
    CSS_NAMES, CSS_LAB_ARRAY,
)

log = logging.getLogger("copalette_css_clothing")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        face, cloth, labels = [b.to(device, non_blocking=True) for b in batch]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(face, cloth)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def top_k_accuracy(logits, labels, k):
    if k >= logits.shape[1]:
        return 1.0
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k_preds == labels[:, None], axis=1)))


def majority_baseline(labels):
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    return float(counts.max()) / len(labels)


def _local_idx_to_global_lab(idx_to_name):
    """For a {local_idx -> css_name} mapping, return an array of shape
    (num_local, 3) of LAB centroids pulled from CSS_LAB_ARRAY."""
    n = len(idx_to_name)
    out = np.zeros((n, 3), dtype=np.float64)
    for local_idx, css_name in idx_to_name.items():
        try:
            global_idx = CSS_NAMES.index(css_name)
        except ValueError:
            # Unknown name — fall back to gray-ish neutral so the row at
            # least exists in the table.
            out[int(local_idx)] = np.array([50.0, 0.0, 0.0])
            continue
        out[int(local_idx)] = CSS_LAB_ARRAY[global_idx]
    return out


def compute_perceptual_delta_e(preds, labels, idx_to_name):
    """For each test sample, look up the LAB centroid of the predicted and
    true CSS class and compute CIEDE2000 (and CIE76) ΔE between them.

    Returns dict with mean / median ΔE00 and ΔE76, plus the per-sample
    arrays so the caller can also dump them to CSV.
    """
    centroids = _local_idx_to_global_lab(idx_to_name)
    pred_lab = centroids[preds.astype(int)]
    true_lab = centroids[labels.astype(int)]

    de00 = delta_e_ciede2000(pred_lab, true_lab)
    de76 = delta_e_cie76(pred_lab, true_lab)

    return {
        "delta_e00_mean": float(np.mean(de00)),
        "delta_e00_median": float(np.median(de00)),
        "delta_e76_mean": float(np.mean(de76)),
        "delta_e76_median": float(np.median(de76)),
    }, de00, de76


def evaluate_css_classification(model, loader, device, idx_to_name,
                                k_values=(1, 3, 5)):
    logits, labels = predict(model, loader, device)
    preds = np.argmax(logits, axis=1)

    result = {
        "top1_accuracy": float(np.mean(preds == labels)),
        "majority_baseline": majority_baseline(labels),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "num_classes": int(logits.shape[1]),
        "num_samples": len(labels),
    }
    for k in k_values:
        result[f"top{k}_accuracy"] = top_k_accuracy(logits, labels, k)

    de_metrics, de00, de76 = compute_perceptual_delta_e(preds, labels, idx_to_name)
    result.update(de_metrics)

    return result, logits, labels, preds, de00, de76


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CSS-COLOR BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def per_css_breakdown(labels, preds, idx_to_name, top_n=20):
    """Per-CSS-color top-1 accuracy, sample count, F1.

    Returns a DataFrame sorted by `n_test` descending, truncated to top_n.
    """
    rows = []
    num_classes = len(idx_to_name)
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(num_classes)),
                            zero_division=0)

    for local_idx, css_name in sorted(idx_to_name.items()):
        mask = labels == local_idx
        n = int(mask.sum())
        if n == 0:
            top1 = 0.0
        else:
            top1 = float(np.mean(preds[mask] == labels[mask]))
        rows.append({
            "css_name": css_name,
            "local_idx": int(local_idx),
            "n_test": n,
            "top1_accuracy": top1,
            "f1": float(per_class_f1[int(local_idx)]),
        })

    df = pd.DataFrame(rows).sort_values("n_test", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df
