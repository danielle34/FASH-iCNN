"""
Evaluation utilities for abstraction-ladder designer classification.

Top-k accuracy, macro F1, baseline.
Self-contained — no imports from other copalette modules.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import f1_score

log = logging.getLogger("abstraction_designer")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        imgs, labels = [b.to(device, non_blocking=True) for b in batch]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


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


def evaluate_classification(model, loader, device, k_values=(1, 3)):
    logits, labels = predict(model, loader, device)
    preds = np.argmax(logits, axis=1)
    result = {
        "top1": float(np.mean(preds == labels)),
        "majority_baseline": majority_baseline(labels),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "num_classes": int(logits.shape[1]),
        "n_test": len(labels),
    }
    for k in k_values:
        result[f"top{k}"] = top_k_accuracy(logits, labels, k)
    return result
