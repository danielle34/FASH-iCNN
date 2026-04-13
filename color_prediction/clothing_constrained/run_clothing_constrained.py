#!/usr/bin/env python3
"""
run_clothing_constrained.py — Main entry point for copalette_clothing_constrained.

Four experiments, all using the SAME EfficientNet-B0 + clothing-only
architecture, all with skip-if-exists checkpointing:

  Exp 1: per-designer BK 9-class
  Exp 2: per-decade BK 9-class
  Exp 3: per-(designer, decade) BK 9-class
  Exp 4: per-designer CSS classification

Each experiment can be skipped independently with --skip_exp1..--skip_exp4.

Usage:
    python run_clothing_constrained.py \
        --csv_path /abs/path/to/copalette_ALL_YEARS.csv \
        --clothing_crops_dir /abs/path/to/clothing_crops/ \
        --output_dir /abs/path/to/copalette_clothing_constrained/outputs \
        --batch_size 64 --seed 42
"""

import os
import sys
import argparse
import logging
import time

import numpy as np
import torch

# Ensure this folder is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    load_and_filter, random_split,
    build_bk_class_mapping, build_css_class_mapping,
    get_train_transform, get_eval_transform, make_loader,
    DECADE_NAMES,
)
from model import ClothingClassifier, count_parameters
from train import train_classifier
from evaluate import (
    evaluate_classifier, save_result_row, load_existing_result,
    log_metrics, log_summary_table,
)


def _slug(text):
    """Make a string safe for use in filenames / CSV condition names."""
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


# ===================================================================
# Args / setup
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CoPalette Clothing Constrained Experiments")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Absolute path to copalette_ALL_YEARS.csv")
    p.add_argument("--clothing_crops_dir", type=str, required=True,
                   help="Absolute path to {image_id}_clothing.jpg directory")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Absolute path to output directory")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_resume", action="store_true",
                   help="Force training from scratch, ignore checkpoints/cached results")
    p.add_argument("--skip_exp1", action="store_true",
                   help="Skip Experiment 1 (per-designer BK)")
    p.add_argument("--skip_exp2", action="store_true",
                   help="Skip Experiment 2 (per-decade BK)")
    p.add_argument("--skip_exp3", action="store_true",
                   help="Skip Experiment 3 (per-designer x decade BK)")
    p.add_argument("--skip_exp4", action="store_true",
                   help="Skip Experiment 4 (per-designer CSS)")
    return p.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "clothing_constrained_experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===================================================================
# Generic per-slice trainer
# ===================================================================
def train_and_eval_slice(
    slice_df, slice_id, label_column, results_csv, ckpt_dir,
    clothing_crops_dir, device, batch_size, no_resume,
    extra_columns=None, css_min_count=10, min_test_samples=20,
    min_total=200,
):
    """
    Generic per-slice train+eval helper used by all 4 experiments.

    `slice_id` becomes the condition name and the checkpoint filename.
    `label_column` is either "c1_bk" (BK 9-class) or "c1_css" (CSS).
    `extra_columns` is a dict of metadata columns to attach to the result row.
    """
    logger = logging.getLogger("Slice")
    extra_columns = extra_columns or {}

    existing = load_existing_result(results_csv, slice_id)
    if existing is not None and not no_resume:
        logger.info(f"  Skipping {slice_id}: already in {results_csv}")
        return

    n_total = len(slice_df)
    if n_total < min_total:
        logger.warning(f"  {slice_id}: only {n_total} chromatic images "
                       f"(< {min_total}), skipping")
        save_result_row(results_csv, {
            "condition": slice_id,
            "skipped_reason": f"fewer_than_{min_total}_images",
            "num_total": n_total,
            **extra_columns,
        })
        return

    train_df, val_df, test_df = random_split(slice_df, seed=42)
    if len(test_df) < min_test_samples:
        logger.warning(f"  {slice_id}: only {len(test_df)} test samples, skipping")
        save_result_row(results_csv, {
            "condition": slice_id,
            "skipped_reason": "test_set_too_small",
            "num_total": n_total,
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df),
            **extra_columns,
        })
        return

    if label_column == "c1_bk":
        class_names, class_to_idx = build_bk_class_mapping()
        remap = None
    else:
        remap, class_names, class_to_idx = build_css_class_mapping(
            train_df, min_count=css_min_count)

    num_classes = len(class_names)
    if num_classes < 2:
        logger.warning(f"  {slice_id}: only {num_classes} classes, skipping")
        save_result_row(results_csv, {
            "condition": slice_id,
            "skipped_reason": "too_few_classes",
            "num_total": n_total,
            "num_classes": num_classes,
            **extra_columns,
        })
        return

    logger.info(f"  {slice_id}: {num_classes} classes, "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_tf = get_train_transform()
    eval_tf = get_eval_transform()

    train_loader = make_loader(
        train_df, clothing_crops_dir, train_tf,
        label_column, class_to_idx, remap=remap,
        batch_size=batch_size, shuffle=True,
    )
    val_loader = make_loader(
        val_df, clothing_crops_dir, eval_tf,
        label_column, class_to_idx, remap=remap,
        batch_size=batch_size, shuffle=False,
    )
    test_loader = make_loader(
        test_df, clothing_crops_dir, eval_tf,
        label_column, class_to_idx, remap=remap,
        batch_size=batch_size, shuffle=False,
    )

    model = ClothingClassifier(num_classes=num_classes)
    logger.info(f"  Parameters: {count_parameters(model):,}")

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{slice_id}.pth")
    best_val_loss, _ = train_classifier(
        model, train_loader, val_loader, device, ckpt_path,
        max_epochs=80, patience=10, no_resume=no_resume,
    )

    metrics = evaluate_classifier(model, test_loader, device, class_names)
    metrics["condition"] = slice_id
    metrics["label_column"] = label_column
    metrics["best_val_loss"] = best_val_loss
    metrics["num_train"] = len(train_df)
    metrics["num_val"] = len(val_df)
    metrics["num_test"] = len(test_df)
    metrics["num_total"] = n_total
    metrics.update(extra_columns)

    log_metrics(metrics, prefix=slice_id)
    save_result_row(results_csv, metrics)
    log_summary_table(results_csv, results_csv.split("/")[-1])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ===================================================================
# Experiments
# ===================================================================
def run_exp1(df, args, device, ckpt_dir):
    logger = logging.getLogger("Exp1")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: per-designer BK 9-class")
    logger.info("=" * 70)
    results_csv = os.path.join(args.output_dir, "per_designer_bk_results.csv")

    designers = sorted(df["designer"].unique())
    for designer in designers:
        sub = df[df["designer"] == designer].reset_index(drop=True)
        slice_id = f"bk_{_slug(designer)}"
        logger.info(f"\n--- Designer: {designer} ({len(sub)} rows) ---")
        train_and_eval_slice(
            sub, slice_id, "c1_bk", results_csv, ckpt_dir,
            args.clothing_crops_dir, device, args.batch_size, args.no_resume,
            extra_columns={"designer": designer},
            min_total=200,
        )


def run_exp2(df, args, device, ckpt_dir):
    logger = logging.getLogger("Exp2")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: per-decade BK 9-class")
    logger.info("=" * 70)
    results_csv = os.path.join(args.output_dir, "per_decade_bk_results.csv")

    for decade in DECADE_NAMES:
        sub = df[df["decade"] == decade].reset_index(drop=True)
        slice_id = f"bk_{_slug(decade)}"
        logger.info(f"\n--- Decade: {decade} ({len(sub)} rows) ---")
        train_and_eval_slice(
            sub, slice_id, "c1_bk", results_csv, ckpt_dir,
            args.clothing_crops_dir, device, args.batch_size, args.no_resume,
            extra_columns={"decade": decade},
            min_total=200,
        )


def run_exp3(df, args, device, ckpt_dir):
    logger = logging.getLogger("Exp3")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: per-(designer x decade) BK 9-class")
    logger.info("=" * 70)
    results_csv = os.path.join(args.output_dir, "per_designer_decade_bk_results.csv")

    designers = sorted(df["designer"].unique())
    for designer in designers:
        for decade in DECADE_NAMES:
            sub = df[(df["designer"] == designer) & (df["decade"] == decade)]
            sub = sub.reset_index(drop=True)
            slice_id = f"bk_{_slug(designer)}__{_slug(decade)}"
            logger.info(f"\n--- {designer} x {decade} ({len(sub)} rows) ---")
            train_and_eval_slice(
                sub, slice_id, "c1_bk", results_csv, ckpt_dir,
                args.clothing_crops_dir, device, args.batch_size, args.no_resume,
                extra_columns={"designer": designer, "decade": decade},
                min_total=100,
            )


def run_exp4(df, args, device, ckpt_dir):
    logger = logging.getLogger("Exp4")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: per-designer CSS")
    logger.info("=" * 70)
    results_csv = os.path.join(args.output_dir, "per_designer_css_results.csv")

    designers = sorted(df["designer"].unique())
    for designer in designers:
        sub = df[df["designer"] == designer].reset_index(drop=True)
        slice_id = f"css_{_slug(designer)}"
        logger.info(f"\n--- Designer: {designer} CSS ({len(sub)} rows) ---")
        train_and_eval_slice(
            sub, slice_id, "c1_css", results_csv, ckpt_dir,
            args.clothing_crops_dir, device, args.batch_size, args.no_resume,
            extra_columns={"designer": designer},
            min_total=200, css_min_count=10,
        )


# ===================================================================
# MAIN
# ===================================================================
def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    set_seed(args.seed)
    device = get_device()

    logger.info("=" * 70)
    logger.info("CoPalette Clothing Constrained Experiments")
    logger.info("=" * 70)
    logger.info(f"CSV:               {args.csv_path}")
    logger.info(f"Clothing crops:    {args.clothing_crops_dir}")
    logger.info(f"Output:            {args.output_dir}")
    logger.info(f"Batch size:        {args.batch_size}")
    logger.info(f"Seed:              {args.seed}")
    logger.info(f"Device:            {device}")
    logger.info(f"No resume:         {args.no_resume}")
    logger.info(f"Skip Exp1:         {args.skip_exp1}")
    logger.info(f"Skip Exp2:         {args.skip_exp2}")
    logger.info(f"Skip Exp3:         {args.skip_exp3}")
    logger.info(f"Skip Exp4:         {args.skip_exp4}")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    df = load_and_filter(
        args.csv_path, args.clothing_crops_dir, args.output_dir,
        num_designers=15, min_designer_count=200,
    )
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s: {len(df)} rows")

    if args.skip_exp1:
        logger.info("Skipping Experiment 1 (--skip_exp1)")
    else:
        try:
            run_exp1(df, args, device, ckpt_dir)
        except Exception as e:
            logger.error(f"Experiment 1 failed: {e}", exc_info=True)

    if args.skip_exp2:
        logger.info("Skipping Experiment 2 (--skip_exp2)")
    else:
        try:
            run_exp2(df, args, device, ckpt_dir)
        except Exception as e:
            logger.error(f"Experiment 2 failed: {e}", exc_info=True)

    if args.skip_exp3:
        logger.info("Skipping Experiment 3 (--skip_exp3)")
    else:
        try:
            run_exp3(df, args, device, ckpt_dir)
        except Exception as e:
            logger.error(f"Experiment 3 failed: {e}", exc_info=True)

    if args.skip_exp4:
        logger.info("Skipping Experiment 4 (--skip_exp4)")
    else:
        try:
            run_exp4(df, args, device, ckpt_dir)
        except Exception as e:
            logger.error(f"Experiment 4 failed: {e}", exc_info=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  per_designer_bk_results.csv")
    logger.info(f"  per_decade_bk_results.csv")
    logger.info(f"  per_designer_decade_bk_results.csv")
    logger.info(f"  per_designer_css_results.csv")
    logger.info(f"  clothing_constrained_dataset.csv")
    logger.info(f"  clothing_constrained_experiment.log")
    logger.info(f"  checkpoints/ckpt_*.pth")


if __name__ == "__main__":
    main()
