#!/usr/bin/env python3
"""
run_hierarchical_lab.py — Main entry point for copalette_hierarchical_lab.

Three-stage hierarchical color prediction:

  Stage 1: Berlin-Kay 9-class prediction
           Loads the upstream multimodal F checkpoint (face + clothing +
           designer) from --full_model_checkpoint and runs it forward.

  Stage 2: CSS color prediction WITHIN the predicted BK family
           One small head per BK family. If --stage2_checkpoint_dir
           contains ckpt_family_<bk>.pth those are loaded. Otherwise the
           heads are trained from scratch on cached features.

  Stage 3: Constrained LAB regression
           A regression head over the frozen face+clothing 2560-dim
           feature predicts an offset that is added to the per-image CSS
           centroid:
               predicted_LAB = css_centroid + tanh(head(features)) * max_offset
           Trained on TRUE CSS centroids, evaluated under both oracle
           (true CSS) and pipeline (predicted CSS) routing.

Four reported conditions:
  1) css_centroid_only       — Stage 2 prediction → its CSS centroid
  2) constrained_oracle      — true BK + true CSS routing + Stage 3 regression
  3) constrained_pipeline    — predicted BK + predicted CSS routing + Stage 3
  4) unconstrained_baseline  — value supplied via --unconstrained_baseline_de00
                                (or NaN if not given)

Usage:
    python run_hierarchical_lab.py \
        --csv_path /abs/path/to/copalette_ALL_YEARS.csv \
        --face_crops_dir /abs/path/to/face_crops/ \
        --clothing_crops_dir /abs/path/to/clothing_crops/ \
        --output_dir /abs/path/to/copalette_hierarchical_lab/outputs \
        --full_model_checkpoint /abs/path/to/ckpt_F_full.pth \
        --stage2_checkpoint_dir /abs/path/to/copalette_hierarchical_color/outputs/checkpoints \
        --batch_size 64 --seed 42
"""

import os
import sys
import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

# Ensure this folder is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colors import (
    CHROMATIC_BK_NAMES, CHROMATIC_BK_TO_IDX, CHROMATIC_BK_IDX_TO_NAME,
    CSS_TO_BK, css_to_lab, css_to_lab_batch,
)
from dataset import (
    load_and_filter, stratified_split,
    get_train_transform, get_eval_transform, make_loader,
)
from model import MultimodalModel, FamilyCSSClassifier, LABRegressor
from train import train_family_css_classifier, train_lab_regressor
from evaluate import (
    extract_features, lab_metrics,
    save_result_row, load_existing_result,
    log_metrics, log_summary_table,
)


# ===================================================================
# Args / setup
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CoPalette Hierarchical LAB Pipeline")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--face_crops_dir", type=str, required=True)
    p.add_argument("--clothing_crops_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--full_model_checkpoint", type=str, required=True,
                   help="Absolute path to multimodal F (face+clothing+designer) checkpoint")
    p.add_argument("--stage1_checkpoint", type=str, default=None,
                   help="(Optional) Stage 1 checkpoint — defaults to --full_model_checkpoint")
    p.add_argument("--stage2_checkpoint_dir", type=str, required=True,
                   help="Directory holding ckpt_family_<bk_name>.pth (loaded if present)")
    p.add_argument("--unconstrained_baseline_de00", type=float, default=15.0,
                   help="ΔE00 to log for the unconstrained baseline (e.g. 15.0)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_resume", action="store_true")
    return p.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "hierarchical_lab_experiment.log")
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
    return torch.device("cpu")


# ===================================================================
# Stage 1: load multimodal F checkpoint
# ===================================================================
def load_multimodal_F(checkpoint_path, num_designers, device):
    logger = logging.getLogger("Stage1")
    logger.info(f"Loading multimodal F checkpoint from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    model = MultimodalModel(num_classes=9, num_designers=num_designers,
                            use_face=True, use_clothing=True, use_designer=True)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unrecognized checkpoint format — expected a state_dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {missing}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    logger.info("Multimodal F loaded and frozen")
    return model


# ===================================================================
# Stage 2: per-family CSS classifiers
# ===================================================================
def build_family_css_class_sets(train_features_pkg, min_count=30):
    """
    For each chromatic BK family, build the list of CSS colors that:
      - map to that BK family (per CSS_TO_BK)
      - appear in the training set with c1_bk == family
      - have count >= min_count

    Returns dict family -> {"class_names": [...], "class_to_idx": {...}}
    """
    bk_true = train_features_pkg["bk_true"]
    css_true = train_features_pkg["true_css"]

    out = {}
    for family in CHROMATIC_BK_NAMES:
        family_idx = CHROMATIC_BK_TO_IDX[family]
        mask = (bk_true == family_idx)
        if mask.sum() == 0:
            out[family] = {"class_names": [], "class_to_idx": {}}
            continue
        family_css = [css_true[i] for i in np.where(mask)[0]
                      if CSS_TO_BK.get(css_true[i]) == family]
        if len(family_css) == 0:
            out[family] = {"class_names": [], "class_to_idx": {}}
            continue
        from collections import Counter
        counts = Counter(family_css)
        kept = [n for n, c in counts.items() if c >= min_count]
        if not kept:
            # Fallback: keep the single most common CSS color in the family
            kept = [counts.most_common(1)[0][0]]
        kept = sorted(kept)
        out[family] = {
            "class_names": kept,
            "class_to_idx": {n: i for i, n in enumerate(kept)},
        }
    return out


def _select_family_features(features_pkg, family, family_class_to_idx):
    """
    Slice the cached features to only the rows whose TRUE BK == family
    AND whose true CSS is in this family's class set. Returns
    (features_tensor, label_tensor).
    """
    family_idx = CHROMATIC_BK_TO_IDX[family]
    bk_true = features_pkg["bk_true"]
    css_true = features_pkg["true_css"]

    keep_idx = [i for i in range(len(css_true))
                if bk_true[i] == family_idx and css_true[i] in family_class_to_idx]
    if not keep_idx:
        return None, None

    feats = features_pkg["fusion_features"][keep_idx]
    labels = np.array([family_class_to_idx[css_true[i]] for i in keep_idx],
                      dtype=np.int64)
    return torch.from_numpy(feats), torch.from_numpy(labels)


def train_or_load_family_classifiers(
    train_pkg, val_pkg, family_class_sets,
    fusion_dim, device, stage2_ckpt_dir, output_ckpt_dir,
    no_resume,
):
    """
    For each chromatic BK family, either load the family classifier
    (preferring --stage2_checkpoint_dir, then our own outputs/checkpoints)
    or train it from scratch on cached features.
    """
    logger = logging.getLogger("Stage2")
    classifiers = {}

    for family in CHROMATIC_BK_NAMES:
        info = family_class_sets[family]
        num_classes = len(info["class_names"])
        if num_classes == 0:
            logger.warning(f"  [{family}] no CSS classes available — skipping")
            classifiers[family] = None
            continue
        logger.info(f"  [{family}] {num_classes} CSS classes: {info['class_names']}")

        external_ckpt = os.path.join(stage2_ckpt_dir, f"ckpt_family_{family}.pth")
        local_ckpt = os.path.join(output_ckpt_dir, f"ckpt_family_{family}.pth")

        loaded = False
        if not no_resume and os.path.exists(external_ckpt):
            ckpt = torch.load(external_ckpt, map_location=device, weights_only=False)
            sd = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
            try:
                m = FamilyCSSClassifier(fusion_dim, num_classes).to(device)
                m.load_state_dict(sd, strict=False)
                m.eval()
                classifiers[family] = m
                logger.info(f"  [{family}] loaded external checkpoint {external_ckpt}")
                loaded = True
            except Exception as e:
                logger.warning(f"  [{family}] external checkpoint load failed: {e}")

        if not loaded:
            tf, tl = _select_family_features(train_pkg, family, info["class_to_idx"])
            vf, vl = _select_family_features(val_pkg, family, info["class_to_idx"])
            if tf is None or vf is None or tf.size(0) == 0 or vf.size(0) == 0:
                logger.warning(f"  [{family}] no training samples — skipping")
                classifiers[family] = None
                continue
            m, _ = train_family_css_classifier(
                family_name=family,
                fusion_dim=fusion_dim,
                train_features=tf, train_labels=tl,
                val_features=vf, val_labels=vl,
                num_classes=num_classes,
                device=device,
                checkpoint_path=local_ckpt,
                no_resume=no_resume,
            )
            m.eval()
            classifiers[family] = m

    return classifiers


def predict_css(features_pkg, family_class_sets, classifiers,
                routing="predicted", device="cpu"):
    """
    For each test image, route to the right family classifier and pick a
    CSS color.

    routing="predicted" -> use bk_pred from features_pkg
    routing="true"      -> use bk_true from features_pkg
    """
    n = features_pkg["fusion_features"].shape[0]
    fusion = torch.from_numpy(features_pkg["fusion_features"]).to(device)
    bk_route = features_pkg[
        "bk_pred" if routing == "predicted" else "bk_true"
    ]

    pred_css = [None] * n
    for family in CHROMATIC_BK_NAMES:
        family_idx = CHROMATIC_BK_TO_IDX[family]
        idx_in_family = np.where(bk_route == family_idx)[0]
        if idx_in_family.size == 0:
            continue
        info = family_class_sets[family]
        clf = classifiers.get(family)
        if clf is None or len(info["class_names"]) == 0:
            # No classifier or no classes — fall back to family CSS == family BK name
            # if it exists, else first CSS color in CSS_TO_BK whose family matches.
            fallback = None
            if family in info["class_names"]:
                fallback = family
            elif info["class_names"]:
                fallback = info["class_names"][0]
            else:
                fallback = family  # raw BK name as best-effort
            for i in idx_in_family:
                pred_css[i] = fallback
            continue

        with torch.no_grad():
            logits = clf(fusion[idx_in_family])
            argmax = logits.argmax(dim=1).cpu().numpy()
        for j, i in enumerate(idx_in_family):
            pred_css[i] = info["class_names"][argmax[j]]

    return pred_css


# ===================================================================
# Stage 3: constrained LAB regression
# ===================================================================
def gather_centroids(css_list):
    return css_to_lab_batch(css_list).astype(np.float32)


def train_stage3_regressor(train_pkg, val_pkg, device, ckpt_path,
                           max_offset, no_resume):
    logger = logging.getLogger("Stage3")
    fc_train = torch.from_numpy(train_pkg["face_clothing_features"])
    fc_val = torch.from_numpy(val_pkg["face_clothing_features"])

    train_centroids = torch.from_numpy(gather_centroids(train_pkg["true_css"]))
    val_centroids = torch.from_numpy(gather_centroids(val_pkg["true_css"]))

    train_lab = torch.from_numpy(train_pkg["true_lab"].astype(np.float32))
    val_lab = torch.from_numpy(val_pkg["true_lab"].astype(np.float32))

    in_dim = fc_train.size(1)
    logger.info(f"  in_dim={in_dim}, max_offset={max_offset}")

    model, _ = train_lab_regressor(
        train_features=fc_train, train_centroids=train_centroids, train_lab=train_lab,
        val_features=fc_val, val_centroids=val_centroids, val_lab=val_lab,
        in_dim=in_dim, device=device, checkpoint_path=ckpt_path,
        max_offset=max_offset, no_resume=no_resume,
    )
    model.eval()
    return model


@torch.no_grad()
def predict_constrained_lab(model, features_pkg, css_list, device, batch_size=2048):
    """Run the LAB regressor over a feature package with the given CSS centroids."""
    fc = torch.from_numpy(features_pkg["face_clothing_features"]).to(device)
    centroids = torch.from_numpy(gather_centroids(css_list)).to(device)
    n = fc.size(0)
    out = []
    for i in range(0, n, batch_size):
        out.append(
            model(fc[i:i + batch_size], centroids[i:i + batch_size]).cpu().numpy()
        )
    return np.concatenate(out, axis=0)


# ===================================================================
# MAIN
# ===================================================================
def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    set_seed(args.seed)
    device = get_device()

    logger.info("=" * 70)
    logger.info("CoPalette Hierarchical LAB Pipeline")
    logger.info("=" * 70)
    logger.info(f"CSV:                 {args.csv_path}")
    logger.info(f"Face crops:          {args.face_crops_dir}")
    logger.info(f"Clothing crops:      {args.clothing_crops_dir}")
    logger.info(f"Output:              {args.output_dir}")
    logger.info(f"Full F checkpoint:   {args.full_model_checkpoint}")
    logger.info(f"Stage 2 ckpt dir:    {args.stage2_checkpoint_dir}")
    logger.info(f"Batch size:          {args.batch_size}")
    logger.info(f"Seed:                {args.seed}")
    logger.info(f"Device:              {device}")
    logger.info(f"No resume:           {args.no_resume}")

    output_ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(output_ckpt_dir, exist_ok=True)

    # ---- Load + filter data ----
    t0 = time.time()
    df, designer_to_idx = load_and_filter(
        args.csv_path, args.face_crops_dir,
        args.clothing_crops_dir, args.output_dir,
    )
    logger.info(f"Dataset loaded in {time.time()-t0:.1f}s: {len(df)} rows, "
                f"{len(designer_to_idx)} designers")
    train_df, val_df, test_df = stratified_split(df, seed=args.seed)

    # ---- Stage 1: load frozen multimodal F ----
    multimodal = load_multimodal_F(
        args.full_model_checkpoint,
        num_designers=len(designer_to_idx),
        device=device,
    )

    # ---- Extract features over train / val / test ----
    eval_tf = get_eval_transform()
    train_loader = make_loader(train_df, args.face_crops_dir, args.clothing_crops_dir,
                               eval_tf, batch_size=args.batch_size, shuffle=False)
    val_loader = make_loader(val_df, args.face_crops_dir, args.clothing_crops_dir,
                             eval_tf, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_df, args.face_crops_dir, args.clothing_crops_dir,
                              eval_tf, batch_size=args.batch_size, shuffle=False)

    logger.info("Extracting features for train/val/test (this is the slow part)...")
    t0 = time.time()
    train_pkg = extract_features(multimodal, train_loader, device)
    logger.info(f"  train: {train_pkg['fusion_features'].shape} "
                f"({time.time()-t0:.1f}s)")
    t0 = time.time()
    val_pkg = extract_features(multimodal, val_loader, device)
    logger.info(f"  val:   {val_pkg['fusion_features'].shape} "
                f"({time.time()-t0:.1f}s)")
    t0 = time.time()
    test_pkg = extract_features(multimodal, test_loader, device)
    logger.info(f"  test:  {test_pkg['fusion_features'].shape} "
                f"({time.time()-t0:.1f}s)")

    fusion_dim = train_pkg["fusion_features"].shape[1]
    logger.info(f"Fusion dim: {fusion_dim}")

    # Stage 1 BK accuracy on test (sanity check)
    bk_acc = float(np.mean(test_pkg["bk_pred"] == test_pkg["bk_true"]))
    logger.info(f"Stage 1 BK 9-class test accuracy: {bk_acc:.4f}")

    # ---- Stage 2: per-family CSS classifiers ----
    logger.info("=" * 70)
    logger.info("STAGE 2: per-family CSS classifiers")
    logger.info("=" * 70)
    family_class_sets = build_family_css_class_sets(train_pkg, min_count=30)
    classifiers = train_or_load_family_classifiers(
        train_pkg, val_pkg, family_class_sets,
        fusion_dim=fusion_dim, device=device,
        stage2_ckpt_dir=args.stage2_checkpoint_dir,
        output_ckpt_dir=output_ckpt_dir,
        no_resume=args.no_resume,
    )

    # Predict CSS for the test set under both routings
    test_css_pred_pipeline = predict_css(
        test_pkg, family_class_sets, classifiers,
        routing="predicted", device=device,
    )
    test_css_pred_oracle = predict_css(
        test_pkg, family_class_sets, classifiers,
        routing="true", device=device,
    )

    # ---- Stage 3: train LAB regressor on TRUE CSS centroids ----
    logger.info("=" * 70)
    logger.info("STAGE 3: constrained LAB regressor")
    logger.info("=" * 70)
    stage3_ckpt = os.path.join(output_ckpt_dir, "ckpt_lab_regression_constrained.pth")
    regressor = train_stage3_regressor(
        train_pkg, val_pkg, device, stage3_ckpt,
        max_offset=(10.0, 15.0, 15.0),
        no_resume=args.no_resume,
    )

    # ---- Evaluation ----
    results_csv = os.path.join(args.output_dir, "stage3_results.csv")

    # Condition 1: CSS centroid only (use the realistic pipeline routing)
    centroids_pipeline = gather_centroids(test_css_pred_pipeline)
    m1 = lab_metrics(centroids_pipeline, test_pkg["true_lab"])
    m1["condition"] = "css_centroid_only"
    m1["routing"] = "pipeline"
    log_metrics(m1, prefix="C1 css_centroid_only")
    save_result_row(results_csv, m1)

    # Condition 2: constrained regression with TRUE BK + TRUE CSS (oracle)
    pred_lab_oracle = predict_constrained_lab(
        regressor, test_pkg, test_pkg["true_css"], device,
    )
    m2 = lab_metrics(pred_lab_oracle, test_pkg["true_lab"])
    m2["condition"] = "constrained_oracle"
    m2["routing"] = "true_bk_true_css"
    log_metrics(m2, prefix="C2 constrained_oracle")
    save_result_row(results_csv, m2)

    # Condition 3: constrained regression with PREDICTED BK + PREDICTED CSS
    pred_lab_pipeline = predict_constrained_lab(
        regressor, test_pkg, test_css_pred_pipeline, device,
    )
    m3 = lab_metrics(pred_lab_pipeline, test_pkg["true_lab"])
    m3["condition"] = "constrained_pipeline"
    m3["routing"] = "pred_bk_pred_css"
    log_metrics(m3, prefix="C3 constrained_pipeline")
    save_result_row(results_csv, m3)

    # Condition 4: unconstrained baseline (logged from CLI)
    m4 = {
        "condition": "unconstrained_baseline",
        "routing": "global_regression_external",
        "delta_e_cie76_mean":   float("nan"),
        "delta_e_cie76_median": float("nan"),
        "delta_e_ciede2000_mean":   float(args.unconstrained_baseline_de00),
        "delta_e_ciede2000_median": float("nan"),
        "bk_accuracy_from_lab": float("nan"),
        "num_samples": int(test_pkg["true_lab"].shape[0]),
    }
    log_metrics(m4, prefix="C4 unconstrained_baseline")
    save_result_row(results_csv, m4)

    log_summary_table(results_csv, "Stage 3 results")

    # Pipeline comparison CSV — same four rows but in a stable column order
    comp_csv = os.path.join(args.output_dir, "pipeline_comparison.csv")
    rows = [m1, m2, m3, m4]
    cols = ["condition", "routing", "delta_e_ciede2000_mean", "delta_e_ciede2000_median",
            "delta_e_cie76_mean", "delta_e_cie76_median", "bk_accuracy_from_lab",
            "num_samples"]
    pd.DataFrame(rows)[cols].to_csv(comp_csv, index=False)
    logger.info(f"Saved pipeline comparison to {comp_csv}")

    # Sanity-check key deltas
    if not np.isnan(m1["delta_e_ciede2000_mean"]) and not np.isnan(m3["delta_e_ciede2000_mean"]):
        delta_p = m1["delta_e_ciede2000_mean"] - m3["delta_e_ciede2000_mean"]
        logger.info(f"  pipeline ΔE00 improvement vs CSS centroid only: {delta_p:+.4f}")
    if not np.isnan(m2["delta_e_ciede2000_mean"]) and not np.isnan(m3["delta_e_ciede2000_mean"]):
        oracle_gap = m3["delta_e_ciede2000_mean"] - m2["delta_e_ciede2000_mean"]
        logger.info(f"  pipeline vs oracle ΔE00 gap (cost of imperfect routing): {oracle_gap:+.4f}")
    if args.unconstrained_baseline_de00 is not None and not np.isnan(m3["delta_e_ciede2000_mean"]):
        vs_unconstr = args.unconstrained_baseline_de00 - m3["delta_e_ciede2000_mean"]
        logger.info(f"  pipeline ΔE00 improvement vs unconstrained baseline: {vs_unconstr:+.4f}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL DONE")
    logger.info("=" * 70)
    logger.info(f"Results:   {results_csv}")
    logger.info(f"Compare:   {comp_csv}")
    logger.info(f"Stage3 ckpt: {stage3_ckpt}")


if __name__ == "__main__":
    main()
