"""
Evaluation utilities for hierarchical color experiments.

Stage 1 BK inference (4-tuple loaders), Stage 2 family CSS inference
(3-tuple loaders), top-k accuracy, macro F1, baseline.
Self-contained — no imports from other copalette modules.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import f1_score

log = logging.getLogger("hierarchical")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: multimodal BK predictions
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_stage1(model, loader, device):
    """Run multimodal BK model on a loader. Returns (preds, labels) numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        face, cloth, des, labels = [b.to(device, non_blocking=True) for b in batch]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(face, cloth, des)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: family CSS predictions
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_family(model, loader, device):
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


def evaluate_family(model, loader, device, k_values=(1, 3, 5)):
    logits, labels = predict_family(model, loader, device)
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
    return result, logits, labels, preds
