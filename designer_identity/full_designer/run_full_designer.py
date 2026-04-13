#!/usr/bin/env python3
"""
Full multimodal face + clothing designer classification.

Research question: Using BOTH face AND clothing crop together, how accurately
can we predict which of 15 fashion houses dressed a model?

Three conditions (all → 15-class designer label, stratified random split):
  A: face only
  B: clothing only
  C: face + clothing (the headline fusion condition)

Self-contained — no imports from other copalette modules.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from dataset import (
    load_and_preprocess, stratified_split_70_15_15, make_loader,
    train_transform, eval_transform,
)
from model import DualCropModel, count_parameters
from train import make_optimizer, train_model
from evaluate import evaluate_classification, plot_confusion_matrix


CONDITIONS = [
    {"id": "A", "name": "A_face_only",     "use_face": True,  "use_clothing": False},
    {"id": "B", "name": "B_clothing_only", "use_face": False, "use_clothing": True},
    {"id": "C", "name": "C_face_clothing", "use_face": True,  "use_clothing": True},
]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log = logging.getLogger("copalette_full_designer")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log_path = os.path.join(output_dir, "full_designer_experiment.log")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def condition_already_done(csv_path, condition_name):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        return condition_name in df["condition"].values
    except Exception:
        return False


def append_result(result_dict, csv_path):
    df_new = pd.DataFrame([result_dict])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)


def print_running_summary(csv_path, log):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
    cols = [c for c in ["condition", "use_face", "use_clothing", "fusion_dim",
                        "top1_accuracy", "top3_accuracy", "majority_baseline",
                        "macro_f1", "num_samples"]
            if c in df.columns]
    log.info(f"\n  Running results:")
    log.info(f"  {df[cols].to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_condition(cond, train_df, val_df, test_df,
                      face_dir, clothing_dir, num_classes, class_names,
                      batch_size, ckpt_dir, output_dir, device, no_resume, log):
    condition_name = cond["name"]
    log.info(f"  use_face={cond['use_face']}, use_clothing={cond['use_clothing']}")
    log.info(f"  Splits: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_loader = make_loader(train_df, face_dir, clothing_dir, train_transform,
                               batch_size, shuffle=True,
                               use_face=cond["use_face"], use_clothing=cond["use_clothing"])
    val_loader = make_loader(val_df, face_dir, clothing_dir, eval_transform,
                             batch_size, shuffle=False,
                             use_face=cond["use_face"], use_clothing=cond["use_clothing"])
    test_loader = make_loader(test_df, face_dir, clothing_dir, eval_transform,
                              batch_size, shuffle=False,
                              use_face=cond["use_face"], use_clothing=cond["use_clothing"])

    model = DualCropModel(num_classes=num_classes,
                          use_face=cond["use_face"], use_clothing=cond["use_clothing"])
    log.info(f"  Model params: {count_parameters(model):,} | fusion_dim={model.fusion_dim}")

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{condition_name}.pth")
    optimizer = make_optimizer(model)

    best_val_loss, _ = train_model(
        model, train_loader, val_loader, optimizer, device,
        max_epochs=100, patience=15, checkpoint_path=ckpt_path,
        no_resume=no_resume,
    )

    metrics, _, labels, preds = evaluate_classification(
        model, test_loader, device, k_values=(1, 3),
    )

    # Per-designer F1
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(num_classes)),
                            zero_division=0)
    per_class_f1_dict = {f"f1_{class_names[i]}": float(per_class_f1[i])
                         for i in range(num_classes)}

    result = {
        "condition": condition_name,
        "condition_id": cond["id"],
        "use_face": cond["use_face"],
        "use_clothing": cond["use_clothing"],
        "fusion_dim": model.fusion_dim,
    }
    result.update(metrics)
    result["best_val_loss"] = best_val_loss
    result.update(per_class_f1_dict)

    log.info(f"  Result: top1={metrics['top1_accuracy']:.3f}, "
             f"top3={metrics['top3_accuracy']:.3f}, "
             f"baseline={metrics['majority_baseline']:.3f}, "
             f"macro_f1={metrics['macro_f1']:.3f}")

    cm_path = os.path.join(output_dir, f"confusion_matrix_{condition_name}.png")
    plot_confusion_matrix(labels, preds, class_names, cm_path,
                          title=f"Designer Confusion — {condition_name}")

    del model, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full multimodal face + clothing designer classification",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--face_crops_dir", type=str, required=True)
    parser.add_argument("--clothing_crops_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    args.csv_path = os.path.abspath(args.csv_path)
    args.face_crops_dir = os.path.abspath(args.face_crops_dir)
    args.clothing_crops_dir = os.path.abspath(args.clothing_crops_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log = setup_logging(args.output_dir)
    log.info("=" * 80)
    log.info("COPALETTE FULL DESIGNER CLASSIFICATION")
    log.info("=" * 80)
    log.info(f"CSV: {args.csv_path}")
    log.info(f"Face crops: {args.face_crops_dir}")
    log.info(f"Clothing crops: {args.clothing_crops_dir}")
    log.info(f"Output: {args.output_dir}")

    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    t_total = time.time()
    df, designer_to_idx = load_and_preprocess(
        args.csv_path, args.face_crops_dir, args.clothing_crops_dir, args.output_dir,
    )
    num_classes = len(designer_to_idx)
    class_names = sorted(designer_to_idx.keys())
    log.info(f"Dataset: {len(df):,} images, {num_classes} designers")

    train_df, val_df, test_df = stratified_split_70_15_15(
        df, strat_col="designer", seed=args.seed,
    )
    log.info(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    results_csv = os.path.join(args.output_dir, "full_designer_results.csv")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for cond in CONDITIONS:
        log.info(f"\n{'=' * 80}")
        log.info(f"CONDITION {cond['id']}: {cond['name']}")
        log.info("=" * 80)

        if not args.no_resume and condition_already_done(results_csv, cond["name"]):
            log.info(f"  Already done, skipping.")
            print_running_summary(results_csv, log)
            continue

        result = run_one_condition(
            cond, train_df, val_df, test_df,
            args.face_crops_dir, args.clothing_crops_dir,
            num_classes, class_names,
            args.batch_size, ckpt_dir, args.output_dir, device, args.no_resume, log,
        )
        if result is not None:
            append_result(result, results_csv)
            print_running_summary(results_csv, log)

    log.info(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
    log.info("All experiments complete.")


if __name__ == "__main__":
    main()
