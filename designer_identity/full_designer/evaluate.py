"""
Evaluation utilities for dual-stream crop color experiments.

Top-k accuracy, macro F1, baseline, confusion matrix plot.
Self-contained — no imports from other copalette modules.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

log = logging.getLogger("copalette_full_designer")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        face_imgs, cloth_imgs, labels = [
            b.to(device, non_blocking=True) for b in batch
        ]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(face_imgs, cloth_imgs)
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


def evaluate_classification(model, loader, device, k_values=(1, 3, 5)):
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
    return result, logits, labels, preds


def plot_confusion_matrix(labels, preds, class_names, output_path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm_norm[i, j]
            if v > 0.005:
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrix to {output_path}")
