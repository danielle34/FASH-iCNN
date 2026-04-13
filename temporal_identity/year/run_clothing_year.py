#!/usr/bin/env python3
"""
Clothing-only year prediction (1991-2024, 34-class) and decade comparison.

Research question: Can a clothing-crop CNN predict the specific year of a
Vogue runway image (34-class)? How does per-year accuracy compare to the
88.6% per-decade result from copalette_clothing_decade?

Three conditions:
  1. clothing -> year (34-class), random stratified split
  2. clothing -> year (34-class), temporal split (train ≤2013 / val 14-16 / test ≥17)
  3. clothing -> decade (4-class), random stratified split (direct comparison)

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
    load_and_preprocess, stratified_split_70_15_15, split_temporal,
    make_loader, train_transform, eval_transform,
    YEAR_MIN, YEAR_MAX, NUM_YEARS, NUM_DECADES, DECADE_LABELS,
)
from model import YearClassifier, count_parameters
from train import make_optimizer, train_model
from evaluate import (
    evaluate_year, evaluate_decade, per_year_breakdown,
    plot_confusion_matrix, plot_per_year_bar_chart,
)


CONDITIONS = [
    {"id": "1", "name": "1_year_random",   "task": "year",   "split": "random"},
    {"id": "2", "name": "2_year_temporal", "task": "year",   "split": "temporal"},
    {"id": "3", "name": "3_decade_random", "task": "decade", "split": "random"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log = logging.getLogger("clothing_year")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log_path = os.path.join(output_dir, "year_experiment.log")
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
    cols = [c for c in ["condition", "task", "split", "num_classes",
                        "top1_accuracy", "top3_accuracy", "majority_baseline",
                        "macro_f1", "mae_years", "adjacent_acc_1",
                        "adjacent_acc_2", "num_samples"]
            if c in df.columns]
    log.info(f"\n  Running results:")
    log.info(f"  {df[cols].to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_condition(cond, df, clothing_dir, batch_size, ckpt_dir,
                      output_dir, device, no_resume, log):
    condition_name = cond["name"]
    task = cond["task"]
    split_kind = cond["split"]

    if split_kind == "random":
        train_df, val_df, test_df = stratified_split_70_15_15(
            df, strat_col="year", seed=42,
        )
    else:
        train_df, val_df, test_df = split_temporal(df)

    log.info(f"  task={task}, split={split_kind}")
    log.info(f"  Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    if task == "year":
        label_col = "year_label"
        num_classes = NUM_YEARS
    else:
        label_col = "decade_label"
        num_classes = NUM_DECADES

    train_loader = make_loader(train_df, clothing_dir, train_transform,
                               label_col, batch_size, shuffle=True)
    val_loader = make_loader(val_df, clothing_dir, eval_transform,
                             label_col, batch_size, shuffle=False)
    test_loader = make_loader(test_df, clothing_dir, eval_transform,
                              label_col, batch_size, shuffle=False)

    model = YearClassifier(num_classes=num_classes)
    log.info(f"  Model params: {count_parameters(model):,}")

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{condition_name}.pth")
    optimizer = make_optimizer(model)

    best_val_loss, _ = train_model(
        model, train_loader, val_loader, optimizer, device,
        max_epochs=100, patience=15, checkpoint_path=ckpt_path,
        no_resume=no_resume,
    )

    if task == "year":
        metrics, _, labels, preds = evaluate_year(
            model, test_loader, device, num_classes, year_min=YEAR_MIN,
        )
        log.info(f"  Result: top1={metrics['top1_accuracy']:.3f}, "
                 f"top3={metrics['top3_accuracy']:.3f}, "
                 f"top5={metrics['top5_accuracy']:.3f}, "
                 f"baseline={metrics['majority_baseline']:.3f}, "
                 f"MAE={metrics['mae_years']:.2f}y, "
                 f"adj±1={metrics['adjacent_acc_1']:.3f}, "
                 f"adj±2={metrics['adjacent_acc_2']:.3f}")

        # Save per-year breakdown for the random condition (Condition 1)
        if condition_name == "1_year_random":
            per_year_df = per_year_breakdown(labels, preds, YEAR_MIN, NUM_YEARS)
            per_year_csv = os.path.join(output_dir, "per_year_accuracy.csv")
            per_year_df.to_csv(per_year_csv, index=False)
            log.info(f"  Saved per-year breakdown to {per_year_csv}")
            plot_per_year_bar_chart(
                per_year_df,
                os.path.join(output_dir, "per_year_accuracy_chart.png"),
            )

        # Save confusion matrix per split
        year_names = [str(YEAR_MIN + k) for k in range(NUM_YEARS)]
        cm_path = os.path.join(output_dir, f"confusion_matrix_{split_kind}.png")
        plot_confusion_matrix(
            labels, preds, year_names, cm_path,
            title=f"Year confusion ({split_kind} split)",
        )
    else:
        metrics, _, labels, preds = evaluate_decade(
            model, test_loader, device, num_classes,
        )
        log.info(f"  Result: top1={metrics['top1_accuracy']:.3f}, "
                 f"baseline={metrics['majority_baseline']:.3f}, "
                 f"macro_f1={metrics['macro_f1']:.3f}")

    result = {
        "condition": condition_name,
        "condition_id": cond["id"],
        "task": task,
        "split": split_kind,
    }
    result.update(metrics)
    result["best_val_loss"] = best_val_loss

    del model, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Clothing-only year/decade prediction",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--clothing_crops_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_condition1", action="store_true")
    parser.add_argument("--skip_condition2", action="store_true")
    parser.add_argument("--skip_condition3", action="store_true")
    args = parser.parse_args()

    args.csv_path = os.path.abspath(args.csv_path)
    args.clothing_crops_dir = os.path.abspath(args.clothing_crops_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log = setup_logging(args.output_dir)
    log.info("=" * 80)
    log.info("COPALETTE CLOTHING YEAR / DECADE PREDICTION")
    log.info("=" * 80)
    log.info(f"CSV: {args.csv_path}")
    log.info(f"Clothing crops: {args.clothing_crops_dir}")
    log.info(f"Output: {args.output_dir}")

    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    t_total = time.time()
    df = load_and_preprocess(args.csv_path, args.clothing_crops_dir, args.output_dir)
    log.info(f"Dataset: {len(df):,} images, "
             f"{df['year'].nunique()} unique years, {NUM_DECADES} decades")

    results_csv = os.path.join(args.output_dir, "year_results.csv")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    skip_flags = {
        "1": args.skip_condition1,
        "2": args.skip_condition2,
        "3": args.skip_condition3,
    }

    for cond in CONDITIONS:
        log.info(f"\n{'=' * 80}")
        log.info(f"CONDITION {cond['id']}: {cond['name']}")
        log.info("=" * 80)

        if skip_flags[cond["id"]]:
            log.info(f"  --skip_condition{cond['id']} flag set, skipping.")
            continue

        if not args.no_resume and condition_already_done(results_csv, cond["name"]):
            log.info(f"  Already done, skipping.")
            print_running_summary(results_csv, log)
            continue

        result = run_one_condition(
            cond, df, args.clothing_crops_dir,
            args.batch_size, ckpt_dir, args.output_dir, device, args.no_resume, log,
        )
        if result is not None:
            append_result(result, results_csv)
            print_running_summary(results_csv, log)

    log.info(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
    log.info("All experiments complete.")


if __name__ == "__main__":
    main()
