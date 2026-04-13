"""
Evaluation utilities for copalette_clothing_decade.
Top-1 accuracy, macro F1, per-class F1, majority baseline, confusion matrix.
"""

import os
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_classifier(model, loader, device, class_names):
    model.eval()
    use_amp = device.type == "cuda"
    all_preds, all_labels = [], []

    for img, label in loader:
        img = img.to(device)
        if use_amp:
            with autocast():
                logits = model(img)
        else:
            logits = model(img)
        all_preds.append(logits.float().argmax(dim=1).cpu().numpy())
        all_labels.append(label.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    num_classes = len(class_names)

    top1 = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class_f1_arr = f1_score(labels, preds, average=None,
                                labels=list(range(num_classes)),
                                zero_division=0)
    per_class_f1 = {f"f1_{class_names[i]}": float(per_class_f1_arr[i])
                    for i in range(num_classes)}

    label_counts = Counter(labels.tolist())
    majority_class = max(label_counts, key=label_counts.get)
    majority_acc = label_counts[majority_class] / len(labels)

    metrics = {
        "top1_accuracy": float(top1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "majority_baseline": float(majority_acc),
        "random_baseline": 1.0 / num_classes,
        "num_classes": num_classes,
        "num_test_samples": int(len(labels)),
    }
    metrics.update(per_class_f1)
    return metrics, labels, preds


def plot_confusion_matrix(labels, preds, class_names, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {save_path}")


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
    logger.info(f"Saved result to {results_csv}: {row_dict.get('condition', 'unknown')}")


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
