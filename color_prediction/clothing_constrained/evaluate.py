"""
Evaluation utilities for copalette_clothing_constrained.
Top-1/3/5 accuracy, macro F1, majority baseline, results CSV I/O.
"""

import os
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score,
)
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_classifier(model, loader, device, class_names):
    model.eval()
    use_amp = device.type == "cuda"
    all_preds, all_labels, all_logits = [], [], []

    for img, label in loader:
        img = img.to(device)
        if use_amp:
            with autocast():
                logits = model(img)
        else:
            logits = model(img)
        logits_f = logits.float()
        all_logits.append(logits_f.cpu().numpy())
        all_preds.append(logits_f.argmax(dim=1).cpu().numpy())
        all_labels.append(label.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    num_classes = len(class_names)

    top1 = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
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
    if num_classes >= 3:
        metrics["top3_accuracy"] = float(top_k_accuracy_score(
            labels, logits, k=3, labels=list(range(num_classes))))
    if num_classes >= 5:
        metrics["top5_accuracy"] = float(top_k_accuracy_score(
            labels, logits, k=5, labels=list(range(num_classes))))

    return metrics


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
