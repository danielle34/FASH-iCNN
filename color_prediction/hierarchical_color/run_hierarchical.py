#!/usr/bin/env python3
"""
Hierarchical color prediction: BK family -> CSS within family.

Research question: Does predicting color hierarchically (first BK9 family,
then specific CSS color within that family) outperform flat direct CSS
prediction?

Pipeline:
  Stage 1 — Load pretrained F_full multimodal model (face + clothing +
            designer -> 9 BK classes), freeze it, run inference to get
            predicted BK labels for every image.
  Stage 2 — For each of 9 BK families, train a separate face+clothing
            CSS classifier using ONLY images whose TRUE BK family matches.
            Backbones initialized from external checkpoints, fine-tuned at
            lr=5e-5. Skip families with <50 images. Drop CSS classes with
            <10 training examples within the family.
  Eval —    Three conditions:
              1. Oracle hierarchical: route by TRUE BK -> family model
              2. Pipeline hierarchical: route by PREDICTED BK -> family model
              3. Flat CSS baseline (read from copalette_noblack expB)

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
    make_multimodal_loader, make_family_loader,
    train_transform, eval_transform,
    CHROMATIC_BK_NAMES, CHROMATIC_BK_TO_IDX, CHROMATIC_BK_IDX_TO_NAME,
)
from model import (
    MultimodalBKModel, FamilyCSSModel,
    load_stage1_checkpoint, init_family_backbones, count_parameters,
)
from train import make_family_optimizer, train_family_model
from evaluate import predict_stage1, predict_family, evaluate_family


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT EXTERNAL CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_STAGE1_CKPT = "/home/morayo/copalette/copalette_multimodal_color/outputs/checkpoints/ckpt_F_full.pth"
DEFAULT_FACE_CKPT = "/home/morayo/copalette/copalette_noblack/outputs/ckpt_A5_random_face_only.pth"
DEFAULT_CLOTHING_CKPT = "/home/morayo/copalette/copalette_multimodal_color/outputs/checkpoints/ckpt_B_clothing_only.pth"
DEFAULT_FLAT_BASELINE_CSV = "/home/morayo/copalette/copalette_noblack/outputs/expB_results.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log = logging.getLogger("hierarchical")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log_path = os.path.join(output_dir, "hierarchical_experiment.log")
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


def append_result(result_dict, csv_path):
    df_new = pd.DataFrame([result_dict])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)


def condition_already_done(csv_path, condition_name, key="condition"):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        return condition_name in df[key].values
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: BK PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage1(df, face_dir, clothing_dir, num_designers,
               stage1_ckpt, batch_size, output_dir, device, no_resume, log):
    log.info("=" * 80)
    log.info("STAGE 1: Multimodal BK predictions (frozen)")
    log.info("=" * 80)

    pred_csv = os.path.join(output_dir, "stage1_predictions.csv")
    if not no_resume and os.path.exists(pred_csv):
        log.info(f"  Stage 1 predictions already exist at {pred_csv}, loading.")
        return pd.read_csv(pred_csv)

    if not os.path.exists(stage1_ckpt):
        log.error(f"  Stage 1 checkpoint not found: {stage1_ckpt}")
        return None

    num_bk_classes = len(CHROMATIC_BK_NAMES)
    model = MultimodalBKModel(num_classes=num_bk_classes, num_designers=num_designers)
    log.info(f"  Model params: {count_parameters(model):,}")
    model = load_stage1_checkpoint(model, stage1_ckpt, device, log)
    model.to(device)
    model.eval()

    loader = make_multimodal_loader(df, face_dir, clothing_dir, eval_transform,
                                    batch_size, shuffle=False)

    log.info(f"  Running inference on {len(df):,} images...")
    preds, labels = predict_stage1(model, loader, device)

    out = pd.DataFrame({
        "image_id": df["image_id"].values,
        "designer": df["designer"].values,
        "true_bk_idx": labels,
        "true_bk_name": [CHROMATIC_BK_IDX_TO_NAME[int(l)] for l in labels],
        "predicted_bk_idx": preds,
        "predicted_bk_name": [CHROMATIC_BK_IDX_TO_NAME[int(p)] for p in preds],
        "correct": (preds == labels).astype(int),
    })
    out.to_csv(pred_csv, index=False)

    acc = float(np.mean(preds == labels))
    log.info(f"  Stage 1 BK accuracy on full dataset: {acc:.4f}")
    log.info(f"  Saved Stage 1 predictions to {pred_csv}")

    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: PER-FAMILY CSS MODELS
# ═══════════════════════════════════════════════════════════════════════════════

MIN_FAMILY_SIZE = 50
MIN_CSS_PER_FAMILY = 10


def run_stage2(df, face_dir, clothing_dir, face_ckpt, clothing_ckpt,
               batch_size, seed, output_dir, device, no_resume, log):
    log.info("=" * 80)
    log.info("STAGE 2: Per-family CSS classifiers")
    log.info("=" * 80)

    csv_path = os.path.join(output_dir, "per_family_results.csv")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Stratified split — use the same split everywhere so test set is consistent
    train_df, val_df, test_df = stratified_split_70_15_15(
        df, strat_col="c1_berlin_kay", seed=seed,
    )
    log.info(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Per-family training
    family_models = {}  # bk_name -> dict with model, css_idx_to_name, etc.
    for bk_name in CHROMATIC_BK_NAMES:
        log.info(f"\n--- Family: {bk_name} ---")

        train_fam = train_df[train_df["c1_berlin_kay"] == bk_name].copy()
        val_fam = val_df[val_df["c1_berlin_kay"] == bk_name].copy()
        test_fam = test_df[test_df["c1_berlin_kay"] == bk_name].copy()
        total = len(train_fam) + len(val_fam) + len(test_fam)

        if total < MIN_FAMILY_SIZE:
            log.info(f"  Skipping {bk_name}: only {total} total images (<{MIN_FAMILY_SIZE})")
            continue

        # Build CSS class mapping from training subset only
        train_counts = train_fam["c1_css_name"].value_counts()
        valid_css = train_counts[train_counts >= MIN_CSS_PER_FAMILY].index.tolist()
        if len(valid_css) < 2:
            log.info(f"  Skipping {bk_name}: only {len(valid_css)} valid CSS classes")
            continue

        css_to_local = {c: i for i, c in enumerate(sorted(valid_css))}
        local_to_css = {i: c for c, i in css_to_local.items()}
        num_css_classes = len(css_to_local)

        train_fam_f = train_fam[train_fam["c1_css_name"].isin(css_to_local)].copy()
        val_fam_f = val_fam[val_fam["c1_css_name"].isin(css_to_local)].copy()
        test_fam_f = test_fam[test_fam["c1_css_name"].isin(css_to_local)].copy()
        train_fam_f["css_local"] = train_fam_f["c1_css_name"].map(css_to_local)
        val_fam_f["css_local"] = val_fam_f["c1_css_name"].map(css_to_local)
        test_fam_f["css_local"] = test_fam_f["c1_css_name"].map(css_to_local)

        if len(train_fam_f) < 20 or len(val_fam_f) < 2 or len(test_fam_f) < 2:
            log.info(f"  Skipping {bk_name}: insufficient split sizes "
                     f"({len(train_fam_f)}/{len(val_fam_f)}/{len(test_fam_f)})")
            continue

        log.info(f"  {bk_name}: {num_css_classes} CSS classes, "
                 f"{len(train_fam_f)}/{len(val_fam_f)}/{len(test_fam_f)}")

        ckpt_path = os.path.join(ckpt_dir, f"ckpt_family_{bk_name}.pth")

        train_loader = make_family_loader(train_fam_f, face_dir, clothing_dir,
                                          train_transform, "css_local",
                                          batch_size, shuffle=True)
        val_loader = make_family_loader(val_fam_f, face_dir, clothing_dir,
                                        eval_transform, "css_local",
                                        batch_size, shuffle=False)
        test_loader = make_family_loader(test_fam_f, face_dir, clothing_dir,
                                         eval_transform, "css_local",
                                         batch_size, shuffle=False)

        model = FamilyCSSModel(num_classes=num_css_classes)
        log.info(f"  Model params: {count_parameters(model):,}")
        init_family_backbones(model, face_ckpt, clothing_ckpt, device, log)

        optimizer = make_family_optimizer(model, backbone_lr=5e-5, head_lr=1e-3)

        # Skip training only if a checkpoint already exists AND a result row exists
        already_done = (not no_resume
                        and condition_already_done(csv_path, bk_name, key="bk_family")
                        and os.path.exists(ckpt_path))
        if already_done:
            log.info(f"  Already done, loading checkpoint.")
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            model.to(device)
            best_val_loss = float(ck.get("best_val_loss", float("nan")))
        else:
            best_val_loss, _ = train_family_model(
                model, train_loader, val_loader, optimizer, device,
                max_epochs=50, patience=10, checkpoint_path=ckpt_path,
                no_resume=no_resume,
            )

        metrics, _, _, _ = evaluate_family(model, test_loader, device,
                                            k_values=(1, 3, 5))

        if not condition_already_done(csv_path, bk_name, key="bk_family"):
            result = {
                "bk_family": bk_name,
                "num_css_classes": num_css_classes,
                "train_size": len(train_fam_f),
                "val_size": len(val_fam_f),
                "test_size": len(test_fam_f),
                "top1_accuracy": metrics["top1_accuracy"],
                "top3_accuracy": metrics["top3_accuracy"],
                "top5_accuracy": metrics["top5_accuracy"],
                "majority_baseline": metrics["majority_baseline"],
                "macro_f1": metrics["macro_f1"],
                "best_val_loss": best_val_loss,
            }
            append_result(result, csv_path)

        log.info(f"  {bk_name}: top1={metrics['top1_accuracy']:.3f}, "
                 f"top3={metrics['top3_accuracy']:.3f}, "
                 f"top5={metrics['top5_accuracy']:.3f}")

        family_models[bk_name] = {
            "model": model,
            "css_to_local": css_to_local,
            "local_to_css": local_to_css,
            "test_df": test_fam_f,
            "test_loader": test_loader,
            "metrics": metrics,
        }

    return family_models, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_oracle_hierarchical(family_models, test_df, log):
    """Condition 1: route by TRUE BK family. Aggregate per-image top-k correctness."""
    total = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    all_true_css = []
    all_pred_css = []

    for bk_name, fam in family_models.items():
        logits, labels = predict_family(fam["model"], fam["test_loader"],
                                        next(fam["model"].parameters()).device)
        local_to_css = fam["local_to_css"]

        preds = np.argmax(logits, axis=1)
        # Top-k correctness
        n = len(labels)
        total += n
        top1_hits += int(np.sum(preds == labels))
        if logits.shape[1] >= 3:
            top3 = np.argsort(logits, axis=1)[:, -3:]
            top3_hits += int(np.sum(np.any(top3 == labels[:, None], axis=1)))
        else:
            top3_hits += n
        if logits.shape[1] >= 5:
            top5 = np.argsort(logits, axis=1)[:, -5:]
            top5_hits += int(np.sum(np.any(top5 == labels[:, None], axis=1)))
        else:
            top5_hits += n

        # Map back to CSS strings for macro F1 across the whole evaluation
        all_true_css.extend([local_to_css[int(l)] for l in labels])
        all_pred_css.extend([local_to_css[int(p)] for p in preds])

    if total == 0:
        return None
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(all_true_css, all_pred_css, average="macro", zero_division=0))

    return {
        "condition": "oracle_hierarchical",
        "top1_accuracy": top1_hits / total,
        "top3_accuracy": top3_hits / total,
        "top5_accuracy": top5_hits / total,
        "num_classes_effective": len(set(all_true_css)),
        "macro_f1": macro_f1,
        "num_samples": total,
    }


def evaluate_pipeline_hierarchical(family_models, test_df, stage1_df, face_dir,
                                   clothing_dir, batch_size, device, log):
    """Condition 2: route by PREDICTED BK family.

    For each image we look up its Stage 1 predicted BK and run that family
    model on it. If the predicted family has no trained model (because we
    skipped that family in Stage 2), the image is counted as wrong.
    """
    # Build image_id -> predicted_bk_name lookup
    stage1_lookup = dict(zip(stage1_df["image_id"], stage1_df["predicted_bk_name"]))

    total = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    all_true_css = []
    all_pred_css = []

    test_df_aligned = test_df.reset_index(drop=True)

    # Group test images by predicted BK family and route to that model
    test_df_aligned = test_df_aligned.copy()
    test_df_aligned["predicted_bk"] = test_df_aligned["image_id"].map(stage1_lookup)

    for predicted_bk in test_df_aligned["predicted_bk"].dropna().unique():
        sub = test_df_aligned[test_df_aligned["predicted_bk"] == predicted_bk]
        n_sub = len(sub)
        if n_sub == 0:
            continue
        total += n_sub

        if predicted_bk not in family_models:
            # No trained family model — count all as wrong
            for _, row in sub.iterrows():
                all_true_css.append(row["c1_css_name"])
                all_pred_css.append("__no_model__")
            continue

        fam = family_models[predicted_bk]
        css_to_local = fam["css_to_local"]
        local_to_css = fam["local_to_css"]

        # Build a loader on this routed subset, with the family's local label
        # space. We need to label rows; if true CSS isn't in the family's
        # vocabulary, the family model can never get it right (true label is
        # outside the predicted family's class set).
        sub = sub.copy()
        sub["css_local"] = sub["c1_css_name"].map(css_to_local)
        in_vocab_mask = sub["css_local"].notna()

        # Out-of-vocabulary rows: counted as wrong
        oov_count = int((~in_vocab_mask).sum())
        for _, row in sub[~in_vocab_mask].iterrows():
            all_true_css.append(row["c1_css_name"])
            all_pred_css.append("__oov__")

        in_vocab = sub[in_vocab_mask].copy()
        if len(in_vocab) == 0:
            continue
        in_vocab["css_local"] = in_vocab["css_local"].astype(int)

        loader = make_family_loader(in_vocab, face_dir, clothing_dir,
                                    eval_transform, "css_local",
                                    batch_size, shuffle=False)
        logits, labels = predict_family(fam["model"], loader, device)
        preds = np.argmax(logits, axis=1)

        top1_hits += int(np.sum(preds == labels))
        if logits.shape[1] >= 3:
            top3 = np.argsort(logits, axis=1)[:, -3:]
            top3_hits += int(np.sum(np.any(top3 == labels[:, None], axis=1)))
        else:
            top3_hits += len(labels)
        if logits.shape[1] >= 5:
            top5 = np.argsort(logits, axis=1)[:, -5:]
            top5_hits += int(np.sum(np.any(top5 == labels[:, None], axis=1)))
        else:
            top5_hits += len(labels)

        all_true_css.extend([local_to_css[int(l)] for l in labels])
        all_pred_css.extend([local_to_css[int(p)] for p in preds])

    if total == 0:
        return None

    from sklearn.metrics import f1_score
    # Compute macro F1 over the union of true CSS labels (predictions like
    # __oov__ and __no_model__ never match any real label, so they correctly
    # count as misses).
    macro_f1 = float(f1_score(all_true_css, all_pred_css, average="macro",
                              zero_division=0))

    return {
        "condition": "pipeline_hierarchical",
        "top1_accuracy": top1_hits / total,
        "top3_accuracy": top3_hits / total,
        "top5_accuracy": top5_hits / total,
        "num_classes_effective": len(set(all_true_css)),
        "macro_f1": macro_f1,
        "num_samples": total,
    }


def load_flat_baseline(flat_baseline_csv, log):
    """Pull the B2 random-split top-1/3/5 from copalette_noblack expB CSV."""
    if not os.path.exists(flat_baseline_csv):
        log.warning(f"  Flat baseline CSV not found: {flat_baseline_csv}")
        return None
    try:
        df = pd.read_csv(flat_baseline_csv)
        candidates = df[df["condition"].astype(str).str.contains("B2", na=False)]
        if len(candidates) == 0:
            candidates = df[df.get("split", "") == "random"]
        if len(candidates) == 0:
            log.warning("  Could not find B2 / random row in flat baseline CSV")
            return None
        row = candidates.iloc[0]
        return {
            "condition": "flat_css_baseline",
            "top1_accuracy": float(row.get("top1_accuracy", row.get("top1", float("nan")))),
            "top3_accuracy": float(row.get("top3_accuracy", row.get("top3", float("nan")))),
            "top5_accuracy": float(row.get("top5_accuracy", row.get("top5", float("nan")))),
            "num_classes_effective": int(row.get("num_classes", -1)),
            "macro_f1": float(row.get("macro_f1", float("nan"))),
            "num_samples": int(row.get("num_samples", -1)),
        }
    except Exception as e:
        log.warning(f"  Failed to parse flat baseline: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def write_comparison_summary(results_csv, output_md, log):
    if not os.path.exists(results_csv):
        return
    df = pd.read_csv(results_csv)
    if len(df) == 0:
        return

    lines = ["# Hierarchical vs Flat CSS Color Prediction\n",
             "Three-condition comparison: oracle hierarchical (true BK route), "
             "pipeline hierarchical (predicted BK route), flat CSS baseline.\n",
             "| Condition | Top-1 | Top-3 | Top-5 | #Classes | Macro F1 | N |",
             "|---|---|---|---|---|---|---|"]
    for _, row in df.iterrows():
        lines.append(
            f"| **{row['condition']}** | "
            f"{float(row['top1_accuracy']):.3f} | "
            f"{float(row['top3_accuracy']):.3f} | "
            f"{float(row['top5_accuracy']):.3f} | "
            f"{int(row['num_classes_effective'])} | "
            f"{float(row['macro_f1']):.3f} | "
            f"{int(row['num_samples'])} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- **oracle vs pipeline**: gap is the cost of Stage 1 BK errors. "
                 "If small, Stage 1 is accurate enough that the hierarchical "
                 "approach works end-to-end.")
    lines.append("- **pipeline vs flat**: this is the headline result. If "
                 "pipeline > flat, decomposing CSS prediction into BK -> CSS "
                 "actually helps. If pipeline < flat, the flat model is better "
                 "at jointly carving up the color space.")
    lines.append("- **oracle vs flat**: the maximum possible improvement from "
                 "hierarchical decomposition, assuming Stage 1 were perfect.\n")

    with open(output_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"  Saved comparison summary to {output_md}")
    log.info("\n" + "\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical color prediction (BK -> CSS within family)",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--face_crops_dir", type=str, required=True)
    parser.add_argument("--clothing_crops_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--face_checkpoint", type=str, default=DEFAULT_FACE_CKPT)
    parser.add_argument("--clothing_checkpoint", type=str, default=DEFAULT_CLOTHING_CKPT)
    parser.add_argument("--flat_baseline_csv", type=str, default=DEFAULT_FLAT_BASELINE_CSV)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    args = parser.parse_args()

    args.csv_path = os.path.abspath(args.csv_path)
    args.face_crops_dir = os.path.abspath(args.face_crops_dir)
    args.clothing_crops_dir = os.path.abspath(args.clothing_crops_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    log = setup_logging(args.output_dir)
    log.info("=" * 80)
    log.info("COPALETTE HIERARCHICAL COLOR EXPERIMENTS")
    log.info("=" * 80)
    log.info(f"CSV: {args.csv_path}")
    log.info(f"Face crops: {args.face_crops_dir}")
    log.info(f"Clothing crops: {args.clothing_crops_dir}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Stage 1 ckpt: {args.stage1_checkpoint}")
    log.info(f"Face init ckpt: {args.face_checkpoint}")
    log.info(f"Clothing init ckpt: {args.clothing_checkpoint}")
    log.info(f"Flat baseline CSV: {args.flat_baseline_csv}")

    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    t_total = time.time()
    df, designer_to_idx = load_and_preprocess(
        args.csv_path, args.face_crops_dir, args.clothing_crops_dir, args.output_dir,
    )
    num_designers = len(designer_to_idx)
    log.info(f"Dataset: {len(df):,} images, {num_designers} designers")

    # Stage 1
    stage1_df = None
    if not args.skip_stage1:
        stage1_df = run_stage1(
            df, args.face_crops_dir, args.clothing_crops_dir, num_designers,
            args.stage1_checkpoint, args.batch_size, args.output_dir,
            device, args.no_resume, log,
        )
    else:
        # Try to load cached Stage 1 predictions
        pred_csv = os.path.join(args.output_dir, "stage1_predictions.csv")
        if os.path.exists(pred_csv):
            stage1_df = pd.read_csv(pred_csv)
            log.info(f"  Loaded cached Stage 1 predictions ({len(stage1_df)} rows)")

    # Stage 2
    family_models = {}
    test_df = None
    if not args.skip_stage2:
        family_models, test_df = run_stage2(
            df, args.face_crops_dir, args.clothing_crops_dir,
            args.face_checkpoint, args.clothing_checkpoint,
            args.batch_size, args.seed, args.output_dir, device, args.no_resume, log,
        )
    else:
        # We still need a test_df aligned with the random split for evaluation
        _, _, test_df = stratified_split_70_15_15(df, strat_col="c1_berlin_kay", seed=args.seed)
        log.info("  Skipped Stage 2 — eval will use no family models (oracle/pipeline disabled)")

    # Evaluation conditions
    log.info("\n" + "=" * 80)
    log.info("EVALUATION CONDITIONS")
    log.info("=" * 80)

    results_csv = os.path.join(args.output_dir, "hierarchical_results.csv")

    # Condition 1: Oracle hierarchical
    if family_models:
        log.info("\n--- Condition 1: oracle_hierarchical ---")
        oracle_result = evaluate_oracle_hierarchical(family_models, test_df, log)
        if oracle_result is not None:
            append_result(oracle_result, results_csv)
            log.info(f"  Oracle: top1={oracle_result['top1_accuracy']:.3f}, "
                     f"top3={oracle_result['top3_accuracy']:.3f}, "
                     f"top5={oracle_result['top5_accuracy']:.3f}")

    # Condition 2: Pipeline hierarchical
    if family_models and stage1_df is not None:
        log.info("\n--- Condition 2: pipeline_hierarchical ---")
        pipeline_result = evaluate_pipeline_hierarchical(
            family_models, test_df, stage1_df,
            args.face_crops_dir, args.clothing_crops_dir,
            args.batch_size, device, log,
        )
        if pipeline_result is not None:
            append_result(pipeline_result, results_csv)
            log.info(f"  Pipeline: top1={pipeline_result['top1_accuracy']:.3f}, "
                     f"top3={pipeline_result['top3_accuracy']:.3f}, "
                     f"top5={pipeline_result['top5_accuracy']:.3f}")

    # Condition 3: Flat CSS baseline (read from external CSV)
    log.info("\n--- Condition 3: flat_css_baseline ---")
    flat_result = load_flat_baseline(args.flat_baseline_csv, log)
    if flat_result is not None:
        append_result(flat_result, results_csv)
        log.info(f"  Flat: top1={flat_result['top1_accuracy']:.3f}, "
                 f"top3={flat_result['top3_accuracy']:.3f}, "
                 f"top5={flat_result['top5_accuracy']:.3f}")

    # Comparison summary
    summary_md = os.path.join(args.output_dir, "comparison_summary.md")
    write_comparison_summary(results_csv, summary_md, log)

    log.info(f"\nTotal runtime: {(time.time() - t_total) / 60:.1f} min")
    log.info("All experiments complete.")


if __name__ == "__main__":
    main()
