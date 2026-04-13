#!/usr/bin/env python3
"""
Face / clothing CSS named-color experiments with perceptual ΔE00.

Research question: Does clothing crop alone predict CSS named colors better
than face+clothing fusion? Same headline question as copalette_crop_color
but at finer (CSS) granularity, plus a critical perceptual metric — even
when the model picks the wrong CSS class, how perceptually close is its
guess to the true class in CIEDE2000 ΔE units?

Five conditions:
  A: clothing-only (color)               — the headline
  B: face-only (color)                   — baseline
  C: face + clothing (color)             — replicates copalette_crop_color cond 4
  D: clothing-only, top-3 swatches view  — same model + ckpt as A, extra reporting
  E: clothing-only (grayscale)           — does structure alone carry CSS signal?

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
    load_and_preprocess, stratified_split_70_15_15,
    build_css_label_mapping, filter_to_valid_css, make_loader,
    train_transform, eval_transform,
    gray_train_transform, gray_eval_transform,
)
from model import DualCropCSSModel, count_parameters
from train import make_optimizer, train_model
from evaluate import (
    evaluate_css_classification, per_css_breakdown,
)


CONDITIONS = [
    {"id": "A", "name": "A_clothing_only",         "use_face": False, "use_clothing": True,  "gray_clothing": False},
    {"id": "B", "name": "B_face_only",             "use_face": True,  "use_clothing": False, "gray_clothing": False},
    {"id": "C", "name": "C_face_clothing",         "use_face": True,  "use_clothing": True,  "gray_clothing": False},
    {"id": "D", "name": "D_clothing_only_swatch",  "use_face": False, "use_clothing": True,  "gray_clothing": False},
    {"id": "E", "name": "E_clothing_only_gray",    "use_face": False, "use_clothing": True,  "gray_clothing": True},
]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log = logging.getLogger("copalette_css_clothing")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log_path = os.path.join(output_dir, "css_clothing_experiment.log")
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
    cols = [c for c in ["condition", "use_face", "use_clothing", "gray_clothing",
                        "num_classes", "top1_accuracy", "top3_accuracy",
                        "top5_accuracy", "majority_baseline", "macro_f1",
                        "delta_e00_mean", "delta_e00_median", "num_samples"]
            if c in df.columns]
    log.info(f"\n  Running results:")
    log.info(f"  {df[cols].to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_condition(cond, train_df, val_df, test_df, name_to_idx, idx_to_name,
                      face_dir, clothing_dir, batch_size, ckpt_dir, output_dir,
                      device, no_resume, log):
    condition_name = cond["name"]
    num_classes = len(name_to_idx)

    log.info(f"  use_face={cond['use_face']}, use_clothing={cond['use_clothing']}, "
             f"gray_clothing={cond['gray_clothing']}")
    log.info(f"  Splits: {len(train_df)}/{len(val_df)}/{len(test_df)}, "
             f"num_classes={num_classes}")

    cloth_train_tf = gray_train_transform if cond["gray_clothing"] else train_transform
    cloth_eval_tf = gray_eval_transform if cond["gray_clothing"] else eval_transform

    train_loader = make_loader(
        train_df, face_dir, clothing_dir, train_transform, cloth_train_tf,
        "css_label", batch_size, shuffle=True,
        use_face=cond["use_face"], use_clothing=cond["use_clothing"],
    )
    val_loader = make_loader(
        val_df, face_dir, clothing_dir, eval_transform, cloth_eval_tf,
        "css_label", batch_size, shuffle=False,
        use_face=cond["use_face"], use_clothing=cond["use_clothing"],
    )
    test_loader = make_loader(
        test_df, face_dir, clothing_dir, eval_transform, cloth_eval_tf,
        "css_label", batch_size, shuffle=False,
        use_face=cond["use_face"], use_clothing=cond["use_clothing"],
    )

    model = DualCropCSSModel(num_classes=num_classes,
                             use_face=cond["use_face"],
                             use_clothing=cond["use_clothing"])
    log.info(f"  Model params: {count_parameters(model):,} | "
             f"fusion_dim={model.fusion_dim}")

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{condition_name}.pth")
    optimizer = make_optimizer(model)

    best_val_loss, _ = train_model(
        model, train_loader, val_loader, optimizer, device,
        max_epochs=100, patience=15, checkpoint_path=ckpt_path,
        no_resume=no_resume,
    )

    metrics, _, labels, preds, de00_per, de76_per = evaluate_css_classification(
        model, test_loader, device, idx_to_name, k_values=(1, 3, 5),
    )

    result = {
        "condition": condition_name,
        "condition_id": cond["id"],
        "use_face": cond["use_face"],
        "use_clothing": cond["use_clothing"],
        "gray_clothing": cond["gray_clothing"],
        "fusion_dim": model.fusion_dim,
    }
    result.update(metrics)
    result["best_val_loss"] = best_val_loss

    log.info(f"  Result: top1={metrics['top1_accuracy']:.3f}, "
             f"top3={metrics['top3_accuracy']:.3f}, "
             f"top5={metrics['top5_accuracy']:.3f}, "
             f"baseline={metrics['majority_baseline']:.3f}, "
             f"macro_f1={metrics['macro_f1']:.3f}")
    log.info(f"  Perceptual: ΔE00 mean={metrics['delta_e00_mean']:.3f}, "
             f"median={metrics['delta_e00_median']:.3f} | "
             f"ΔE76 mean={metrics['delta_e76_mean']:.3f}")

    # Per-CSS breakdown — only for Condition A (the headline)
    if cond["id"] == "A":
        per_df = per_css_breakdown(labels, preds, idx_to_name, top_n=20)
        per_csv = os.path.join(output_dir, "per_css_accuracy.csv")
        per_df.to_csv(per_csv, index=False)
        log.info(f"  Saved per-CSS breakdown to {per_csv}")

    del model, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE ΔE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def write_delta_e_table(results_csv, output_csv, log):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    if len(df) == 0:
        return
    keep = ["condition", "delta_e00_mean", "delta_e00_median",
            "delta_e76_mean", "delta_e76_median", "top1_accuracy",
            "macro_f1", "num_samples"]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(output_csv, index=False)
    log.info(f"  Saved ΔE table to {output_csv}")
    log.info(f"\n{df[keep].to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CSS clothing color experiments with perceptual ΔE00",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--face_crops_dir", type=str, required=True)
    parser.add_argument("--clothing_crops_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    for c in CONDITIONS:
        parser.add_argument(f"--skip_{c['id']}", action="store_true",
                            help=f"Skip condition {c['id']} ({c['name']})")
    args = parser.parse_args()

    args.csv_path = os.path.abspath(args.csv_path)
    args.face_crops_dir = os.path.abspath(args.face_crops_dir)
    args.clothing_crops_dir = os.path.abspath(args.clothing_crops_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log = setup_logging(args.output_dir)
    log.info("=" * 80)
    log.info("COPALETTE CSS CLOTHING EXPERIMENTS")
    log.info("=" * 80)
    log.info(f"CSV: {args.csv_path}")
    log.info(f"Face crops: {args.face_crops_dir}")
    log.info(f"Clothing crops: {args.clothing_crops_dir}")
    log.info(f"Output: {args.output_dir}")

    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    t_total = time.time()
    df = load_and_preprocess(
        args.csv_path, args.face_crops_dir, args.clothing_crops_dir, args.output_dir,
    )

    train_df, val_df, test_df = stratified_split_70_15_15(
        df, strat_col="c1_css_name", seed=args.seed,
    )
    log.info(f"Stratified split: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    name_to_idx = build_css_label_mapping(train_df)
    idx_to_name = {i: n for n, i in name_to_idx.items()}
    log.info(f"CSS classes (>=30 train): {len(name_to_idx)}")

    train_df = filter_to_valid_css(train_df, name_to_idx)
    val_df = filter_to_valid_css(val_df, name_to_idx)
    test_df = filter_to_valid_css(test_df, name_to_idx)
    log.info(f"After CSS filter: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    results_csv = os.path.join(args.output_dir, "css_clothing_results.csv")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for cond in CONDITIONS:
        log.info(f"\n{'=' * 80}")
        log.info(f"CONDITION {cond['id']}: {cond['name']}")
        log.info("=" * 80)

        if getattr(args, f"skip_{cond['id']}"):
            log.info(f"  --skip_{cond['id']} flag set, skipping.")
            continue

        if not args.no_resume and condition_already_done(results_csv, cond["name"]):
            log.info(f"  Already done, skipping.")
            print_running_summary(results_csv, log)
            continue

        result = run_one_condition(
            cond, train_df, val_df, test_df, name_to_idx, idx_to_name,
            args.face_crops_dir, args.clothing_crops_dir,
            args.batch_size, ckpt_dir, args.output_dir, device, args.no_resume, log,
        )
        if result is not None:
            append_result(result, results_csv)
            print_running_summary(results_csv, log)

    # ΔE table
    log.info("\n" + "=" * 80)
    log.info("PERCEPTUAL ΔE TABLE")
    log.info("=" * 80)
    write_delta_e_table(
        results_csv, os.path.join(args.output_dir, "css_delta_e.csv"), log,
    )

    log.info(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
    log.info("All experiments complete.")


if __name__ == "__main__":
    main()
