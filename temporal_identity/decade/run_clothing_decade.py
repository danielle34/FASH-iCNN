#!/usr/bin/env python3
"""
run_clothing_decade.py — Main entry point for copalette_clothing_decade.

Two conditions:
  Condition 1: clothing_only — clothing crop -> decade (4-class)
  Condition 2: face_only     — face crop     -> decade (4-class)

Usage:
    python run_clothing_decade.py \
        --csv_path /abs/path/to/copalette_ALL_YEARS.csv \
        --face_crops_dir /abs/path/to/face_crops/ \
        --clothing_crops_dir /abs/path/to/clothing_crops/ \
        --output_dir /abs/path/to/copalette_clothing_decade/outputs \
        --batch_size 128 --seed 42
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
    load_and_filter, stratified_split,
    get_train_transform, get_eval_transform, make_loader,
    DECADE_NAMES, NUM_DECADES,
)
from model import DecadeModel, count_parameters
from train import train_classifier
from evaluate import (
    evaluate_classifier, plot_confusion_matrix,
    save_result_row, load_existing_result,
    log_metrics, log_summary_table,
)


def parse_args():
    p = argparse.ArgumentParser(description="CoPalette Clothing Decade Prediction")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Absolute path to copalette_ALL_YEARS.csv")
    p.add_argument("--face_crops_dir", type=str, required=True,
                   help="Absolute path to {image_id}_face.jpg directory")
    p.add_argument("--clothing_crops_dir", type=str, required=True,
                   help="Absolute path to {image_id}_clothing.jpg directory")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Absolute path to output directory")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_resume", action="store_true",
                   help="Force training from scratch, ignore checkpoints/cached results")
    return p.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "decade_experiment.log")
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
# Condition runner
# ===================================================================
def run_condition(condition, mode, train_df, val_df, test_df,
                  face_crops_dir, clothing_crops_dir, output_dir,
                  device, batch_size, no_resume):
    logger = logging.getLogger("Decade")
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"CONDITION: {condition} (mode={mode})")
    logger.info("=" * 70)

    results_csv = os.path.join(output_dir, "decade_results.csv")

    existing = load_existing_result(results_csv, condition)
    if existing is not None and not no_resume:
        logger.info(f"  Skipping {condition}: already in {results_csv}")
        return

    train_tf = get_train_transform()
    eval_tf = get_eval_transform()

    train_loader = make_loader(
        train_df, face_crops_dir, clothing_crops_dir, train_tf, mode,
        batch_size=batch_size, shuffle=True,
    )
    val_loader = make_loader(
        val_df, face_crops_dir, clothing_crops_dir, eval_tf, mode,
        batch_size=batch_size, shuffle=False,
    )
    test_loader = make_loader(
        test_df, face_crops_dir, clothing_crops_dir, eval_tf, mode,
        batch_size=batch_size, shuffle=False,
    )

    model = DecadeModel(num_classes=NUM_DECADES)
    logger.info(f"  Parameters: {count_parameters(model):,}")

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{condition}.pth")

    best_val_loss, _ = train_classifier(
        model, train_loader, val_loader, device, ckpt_path,
        max_epochs=80, patience=10, no_resume=no_resume,
    )

    metrics, labels, preds = evaluate_classifier(
        model, test_loader, device, DECADE_NAMES,
    )
    metrics["condition"] = condition
    metrics["mode"] = mode
    metrics["best_val_loss"] = best_val_loss
    metrics["num_train"] = len(train_df)
    metrics["num_val"] = len(val_df)
    metrics["num_test"] = len(test_df)

    log_metrics(metrics, prefix=f"Condition {condition}")
    save_result_row(results_csv, metrics)

    cm_path = os.path.join(output_dir, f"confusion_matrix_{condition}.png")
    plot_confusion_matrix(
        labels, preds, DECADE_NAMES, cm_path,
        title=f"Decade prediction — {condition}",
    )

    log_summary_table(results_csv, "Decade prediction")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ===================================================================
# MAIN
# ===================================================================
def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    set_seed(args.seed)
    device = get_device()

    logger.info("=" * 70)
    logger.info("CoPalette Clothing Decade Experiment")
    logger.info("=" * 70)
    logger.info(f"CSV:               {args.csv_path}")
    logger.info(f"Face crops:        {args.face_crops_dir}")
    logger.info(f"Clothing crops:    {args.clothing_crops_dir}")
    logger.info(f"Output:            {args.output_dir}")
    logger.info(f"Batch size:        {args.batch_size}")
    logger.info(f"Seed:              {args.seed}")
    logger.info(f"Device:            {device}")
    logger.info(f"No resume:         {args.no_resume}")

    t0 = time.time()
    df = load_and_filter(
        args.csv_path, args.face_crops_dir,
        args.clothing_crops_dir, args.output_dir,
    )
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s: {len(df)} rows")

    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    conditions = [
        ("clothing_only", "clothing"),
        ("face_only",     "face"),
    ]

    for cond_name, mode in conditions:
        try:
            run_condition(
                cond_name, mode, train_df, val_df, test_df,
                args.face_crops_dir, args.clothing_crops_dir,
                args.output_dir, device, args.batch_size, args.no_resume,
            )
        except Exception as e:
            logger.error(f"Condition {cond_name} failed: {e}", exc_info=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL CONDITIONS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  decade_results.csv")
    logger.info(f"  dataset.csv")
    logger.info(f"  confusion_matrix_clothing_only.png")
    logger.info(f"  confusion_matrix_face_only.png")
    logger.info(f"  decade_experiment.log")
    logger.info(f"  checkpoints/ckpt_clothing_only.pth, ckpt_face_only.pth")


if __name__ == "__main__":
    main()
