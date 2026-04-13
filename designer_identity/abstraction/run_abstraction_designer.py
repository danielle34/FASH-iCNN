#!/usr/bin/env python3
"""
Clothing-crop abstraction-ladder designer classification.

Four conditions, one EfficientNet-B0 each, all predicting the 15-way
designer label from the SAME clothing crop viewed through four
progressively more abstract representations:

  1. fullcolor  — RGB crop
  2. grayscale  — desaturated crop, 3 channels
  3. silhouette — filled binary mask
  4. edge       — Canny edge map

Pre-generated silhouettes / edges are used if present; otherwise both
are generated on the fly from the clothing crop.

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

from dataset import (
    load_and_preprocess, stratified_split_80_10_10, make_loader,
    train_transform_color, train_transform_mono, eval_transform,
    CONDITIONS, DEFAULT_SILHOUETTE_DIR, DEFAULT_EDGE_DIR,
)
from model import AbstractionClassifier, count_parameters
from train import make_optimizer, train_model
from evaluate import evaluate_classification


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log = logging.getLogger("abstraction_designer")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log_path = os.path.join(output_dir, "abstraction_designer_experiment.log")
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
    cols = [c for c in ["condition", "top1", "top3", "macro_f1",
                        "majority_baseline", "n_test"]
            if c in df.columns]
    log.info(f"\n  Running results:")
    log.info(f"  {df[cols].to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_condition(cond, train_df, val_df, test_df, num_classes,
                      clothing_dir, silhouette_dir, edge_dir,
                      batch_size, ckpt_dir, device, no_resume, log):
    log.info(f"  Condition: {cond}")
    log.info(f"  Splits: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # Grayscale / silhouette / edge skip ColorJitter
    train_tf = train_transform_color if cond == "fullcolor" else train_transform_mono

    train_loader = make_loader(train_df, clothing_dir, silhouette_dir, edge_dir,
                               train_tf, cond, batch_size, shuffle=True)
    val_loader = make_loader(val_df, clothing_dir, silhouette_dir, edge_dir,
                             eval_transform, cond, batch_size, shuffle=False)
    test_loader = make_loader(test_df, clothing_dir, silhouette_dir, edge_dir,
                              eval_transform, cond, batch_size, shuffle=False)

    model = AbstractionClassifier(num_classes=num_classes)
    log.info(f"  Model params: {count_parameters(model):,}")

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{cond}.pth")
    optimizer = make_optimizer(model)

    best_val_loss, _ = train_model(
        model, train_loader, val_loader, optimizer, device,
        max_epochs=100, patience=15, checkpoint_path=ckpt_path,
        no_resume=no_resume,
    )

    metrics = evaluate_classification(model, test_loader, device, k_values=(1, 3))

    result = {
        "condition": cond,
        "top1": metrics["top1"],
        "top3": metrics["top3"],
        "macro_f1": metrics["macro_f1"],
        "majority_baseline": metrics["majority_baseline"],
        "n_test": metrics["n_test"],
        "num_classes": metrics["num_classes"],
        "best_val_loss": best_val_loss,
    }

    log.info(f"  Result: top1={metrics['top1']:.4f}, "
             f"top3={metrics['top3']:.4f}, "
             f"macro_f1={metrics['macro_f1']:.4f}, "
             f"baseline={metrics['majority_baseline']:.4f}")

    del model, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Clothing-crop abstraction-ladder designer classification",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--clothing_crops_dir", type=str,
                        default="/home/morayo/copalette/clothing/")
    parser.add_argument("--silhouette_dir", type=str,
                        default=DEFAULT_SILHOUETTE_DIR)
    parser.add_argument("--edge_crops_dir", type=str, default=DEFAULT_EDGE_DIR)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "outputs"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_condition1", action="store_true",
                        help="Skip fullcolor")
    parser.add_argument("--skip_condition2", action="store_true",
                        help="Skip grayscale")
    parser.add_argument("--skip_condition3", action="store_true",
                        help="Skip silhouette")
    parser.add_argument("--skip_condition4", action="store_true",
                        help="Skip edge")
    args = parser.parse_args()

    args.csv_path = os.path.abspath(args.csv_path)
    args.clothing_crops_dir = os.path.abspath(args.clothing_crops_dir)
    args.silhouette_dir = os.path.abspath(args.silhouette_dir)
    args.edge_crops_dir = os.path.abspath(args.edge_crops_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log = setup_logging(args.output_dir)
    log.info("=" * 80)
    log.info("COPALETTE CLOTHING ABSTRACTION-LADDER DESIGNER CLASSIFICATION")
    log.info("=" * 80)
    log.info(f"CSV: {args.csv_path}")
    log.info(f"Clothing crops: {args.clothing_crops_dir}")
    log.info(f"Silhouette dir (optional): {args.silhouette_dir}")
    log.info(f"Edge dir (optional): {args.edge_crops_dir}")
    log.info(f"Output: {args.output_dir}")

    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    t_total = time.time()
    df, designer_to_idx = load_and_preprocess(
        args.csv_path, args.clothing_crops_dir, args.output_dir,
    )
    num_classes = len(designer_to_idx)
    log.info(f"Dataset: {len(df):,} images, {num_classes} designers")

    train_df, val_df, test_df = stratified_split_80_10_10(
        df, strat_col="designer", seed=args.seed,
    )
    log.info(f"Stratified 80/10/10 split: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    results_csv = os.path.join(args.output_dir, "abstraction_designer_results.csv")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    skip_flags = {
        "fullcolor": args.skip_condition1,
        "grayscale": args.skip_condition2,
        "silhouette": args.skip_condition3,
        "edge": args.skip_condition4,
    }

    for cond in CONDITIONS:
        log.info(f"\n{'=' * 80}")
        log.info(f"CONDITION: {cond}")
        log.info("=" * 80)

        if skip_flags[cond]:
            log.info(f"  skip flag set, skipping.")
            continue

        if not args.no_resume and condition_already_done(results_csv, cond):
            log.info(f"  Already done, skipping.")
            print_running_summary(results_csv, log)
            continue

        try:
            result = run_one_condition(
                cond, train_df, val_df, test_df, num_classes,
                args.clothing_crops_dir, args.silhouette_dir, args.edge_crops_dir,
                args.batch_size, ckpt_dir, device, args.no_resume, log,
            )
            append_result(result, results_csv)
            print_running_summary(results_csv, log)
        except Exception as e:
            log.error(f"  Condition {cond} failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            continue

    log.info(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
    log.info("All experiments complete.")


if __name__ == "__main__":
    main()
