"""
Evaluation utilities for year/decade classification.

Top-k accuracy, majority + random baselines, mean absolute year error,
adjacent accuracies (±1 and ±2 years), per-year F1, confusion matrix plot,
per-year accuracy bar chart.
Self-contained — no imports from other copalette modules.
"""

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, confusion_matrix

log = logging.getLogger("clothing_year")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

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


def adjacent_accuracy(preds, labels, tol):
    """Predicted within ±tol of true (in label-index units, which equal year units)."""
    return float(np.mean(np.abs(preds.astype(int) - labels.astype(int)) <= tol))


def mae_years(preds, labels):
    return float(np.mean(np.abs(preds.astype(int) - labels.astype(int))))


def evaluate_year(model, loader, device, num_classes, year_min=None):
    """Full year evaluation: top-1/3/5, baselines, MAE, adjacent ±1/±2.

    If year_min is provided, MAE/adjacent are reported in actual year units
    (which is the same as label-index units since the year_label encoding is
    just `year - year_min`).
    """
    logits, labels = predict(model, loader, device)
    preds = np.argmax(logits, axis=1)

    result = {
        "top1_accuracy": float(np.mean(preds == labels)),
        "top3_accuracy": top_k_accuracy(logits, labels, 3),
        "top5_accuracy": top_k_accuracy(logits, labels, 5),
        "majority_baseline": majority_baseline(labels),
        "random_baseline": 1.0 / num_classes,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "mae_years": mae_years(preds, labels),
        "adjacent_acc_1": adjacent_accuracy(preds, labels, 1),
        "adjacent_acc_2": adjacent_accuracy(preds, labels, 2),
        "num_classes": num_classes,
        "num_samples": len(labels),
    }
    return result, logits, labels, preds


def evaluate_decade(model, loader, device, num_classes):
    """Compact evaluation for the 4-class decade head."""
    logits, labels = predict(model, loader, device)
    preds = np.argmax(logits, axis=1)
    result = {
        "top1_accuracy": float(np.mean(preds == labels)),
        "top3_accuracy": top_k_accuracy(logits, labels, 3),
        "top5_accuracy": top_k_accuracy(logits, labels, 5),
        "majority_baseline": majority_baseline(labels),
        "random_baseline": 1.0 / num_classes,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "num_classes": num_classes,
        "num_samples": len(labels),
    }
    return result, logits, labels, preds


# ═══════════════════════════════════════════════════════════════════════════════
# PER-YEAR BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def per_year_breakdown(labels, preds, year_min, num_years):
    """Compute per-year top-1, F1, and sample count."""
    rows = []
    for k in range(num_years):
        mask = labels == k
        n = int(mask.sum())
        year = year_min + k
        if n == 0:
            rows.append({"year": year, "year_label": k, "top1_accuracy": 0.0,
                         "f1": 0.0, "n_test": 0})
            continue
        acc = float(np.mean(preds[mask] == labels[mask]))
        rows.append({
            "year": year,
            "year_label": k,
            "top1_accuracy": acc,
            "n_test": n,
        })

    # Compute per-class F1 over the full label range
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(num_years)),
                            zero_division=0)
    for row, f1 in zip(rows, per_class_f1):
        row["f1"] = float(f1)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(labels, preds, class_names, output_path,
                          title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    # For 34 classes, show every label rotated
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_ylabel("True year", fontsize=12)
    ax.set_xlabel("Predicted year", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved confusion matrix to {output_path}")


def plot_per_year_bar_chart(per_year_df, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = per_year_df.sort_values("year").reset_index(drop=True)
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    bars = ax.bar(df["year"].astype(int).astype(str), df["top1_accuracy"].values,
                  color="#4C72B0", edgecolor="#333", linewidth=0.4)

    # Highlight the best and worst years
    best_idx = int(df["top1_accuracy"].idxmax())
    worst_idx = int(df["top1_accuracy"].idxmin())
    bars[best_idx].set_color("#2ca02c")
    bars[worst_idx].set_color("#d62728")

    mean_acc = float(df["top1_accuracy"].mean())
    ax.axhline(y=mean_acc, color="gray", linestyle="--", linewidth=1.0,
               label=f"mean = {mean_acc:.3f}")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Top-1 accuracy", fontsize=12)
    ax.set_title("Per-year top-1 accuracy (clothing-only)", fontsize=14)
    ax.set_ylim(0, max(0.4, df["top1_accuracy"].max() + 0.05))
    plt.xticks(rotation=45, fontsize=8)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved per-year bar chart to {output_path}")
