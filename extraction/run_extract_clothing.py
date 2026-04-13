#!/usr/bin/env python3
"""
Clothing crop extraction from the HuggingFace tonyassi/vogue-runway-top15-512px
dataset using SegFormer (isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing).

For each image:
  1. Run SegFormer to get pixel-level segmentation (0=bg, 1=skin, 2=hair, 3=clothing)
  2. Extract ONLY clothing pixels (label 3)
  3. Place them on a pure white background
  4. Crop tightly to the clothing bounding box with 5% padding
  5. Resize to 512x512 (LANCZOS) and save as JPEG quality 95
  6. Skip images with fewer than 500 clothing pixels

Filtering: Only process image_ids that exist in the input CSV with skip_reason=null
and belong to the top-15 designer list. image_id format must match the existing
face crops in /home/morayo/copalette/face/.

Self-contained — no imports from other copalette modules.
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_NAME = "tonyassi/vogue-runway-top15-512px"
MODEL_NAME = "isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing"

CLOTHING_LABEL = 3
MIN_CLOTHING_PIXELS = 500
PADDING_PCT = 0.05
OUTPUT_SIZE = 512
JPEG_QUALITY = 95
CHECKPOINT_INTERVAL = 500

TOP_15_DESIGNERS = {
    "alexander mcqueen", "armani prive", "balenciaga",
    "calvin klein collection", "chanel", "christian dior",
    "fendi", "gucci", "hermes", "louis vuitton", "prada",
    "ralph lauren", "saint laurent", "valentino", "versace",
}


# ═══════════════════════════════════════════════════════════════════════════════
# build_image_id — must match face crop naming exactly
# ═══════════════════════════════════════════════════════════════════════════════

def build_image_id(label_int, label_names, hf_index):
    """Build image_id from HF dataset label index + global HF dataset position.

    label_names entries look like: 'chanel,fall 1991 ready to wear'
    Output: '{designer}_{season}_{year}_{hf_index:06d}'
    """
    label_str = label_names[label_int]
    parts = label_str.split(',', 1)
    if len(parts) != 2:
        return None
    designer = parts[0].strip().lower().replace(' ', '_')
    show = parts[1].strip()
    tokens = show.split()
    year = None
    season = None
    for i, t in enumerate(tokens):
        if t.isdigit() and len(t) == 4:
            year = t
            season = tokens[0] if i > 0 else 'unknown'
            break
    if year is None:
        return None
    season = season.lower().replace(' ', '_')
    return f"{designer}_{season}_{year}_{hf_index:06d}"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    log = logging.getLogger("clothing_crops")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    log_path = os.path.join(output_dir, "clothing_extraction.log")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class HFImageDataset(Dataset):
    """Wraps the HuggingFace dataset and yields (index, image_id, PIL.Image)."""

    def __init__(self, hf_ds, valid_ids, processor, label_names, start_index=0):
        self.hf_ds = hf_ds
        self.valid_ids = valid_ids
        self.processor = processor
        self.label_names = label_names
        self.start_index = start_index

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        # Always allow index access; the main loop handles skipping
        try:
            row = self.hf_ds[idx]
            label_int = row.get("label", None)
            image = row.get("image", None)
            if label_int is None or image is None:
                return idx, None, None, None

            image_id = build_image_id(label_int, self.label_names, idx)
            if image_id is None:
                return idx, None, None, None

            # Skip if not in valid set
            if image_id not in self.valid_ids:
                return idx, image_id, None, None

            # Convert to RGB PIL image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            image = image.convert("RGB")

            # Pre-process for the model. Returns dict of tensors.
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)

            return idx, image_id, image, pixel_values
        except Exception:
            return idx, None, None, None


def collate_fn(batch):
    """Variable-size collate that handles None entries gracefully."""
    valid = [(i, iid, img, pv) for i, iid, img, pv in batch if pv is not None]
    if not valid:
        # Return a sentinel batch with empty valid lists but full skipped meta
        all_meta = [(i, iid, img) for i, iid, img, pv in batch]
        return ([], [], [], None, all_meta), None

    indices = [v[0] for v in valid]
    image_ids = [v[1] for v in valid]
    images = [v[2] for v in valid]
    pixel_values = torch.stack([v[3] for v in valid], dim=0)

    # Also pass through metadata for skipped items so the main loop can log them
    skipped = [(i, iid, img) for i, iid, img, pv in batch if pv is None]
    return (indices, image_ids, images, pixel_values, skipped), None


# ═══════════════════════════════════════════════════════════════════════════════
# CLOTHING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_clothing_crop(pil_image, clothing_mask):
    """Apply clothing mask to image, crop to bbox+padding, resize to 512x512.

    Returns (output_image, bbox_tuple, clothing_pct, clothing_px_count) or
    (None, None, None, count) if insufficient clothing.
    """
    img_array = np.array(pil_image)  # (H, W, 3)
    h, w = img_array.shape[:2]
    total_px = h * w

    # Resize mask to match original image if needed
    if clothing_mask.shape != (h, w):
        mask_pil = Image.fromarray((clothing_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        clothing_mask = (np.array(mask_pil) > 127)

    clothing_px_count = int(clothing_mask.sum())
    if clothing_px_count < MIN_CLOTHING_PIXELS:
        return None, None, 0.0, clothing_px_count

    clothing_pct = clothing_px_count / total_px

    # White background, copy clothing pixels only
    white_bg = np.ones_like(img_array) * 255
    white_bg[clothing_mask] = img_array[clothing_mask]

    # Bounding box of clothing mask
    rows_any = np.any(clothing_mask, axis=1)
    cols_any = np.any(clothing_mask, axis=0)
    min_row, max_row = np.where(rows_any)[0][[0, -1]]
    min_col, max_col = np.where(cols_any)[0][[0, -1]]

    # Add 5% padding
    bbox_h = max_row - min_row + 1
    bbox_w = max_col - min_col + 1
    pad_h = int(round(bbox_h * PADDING_PCT))
    pad_w = int(round(bbox_w * PADDING_PCT))

    y1 = max(0, min_row - pad_h)
    y2 = min(h - 1, max_row + pad_h)
    x1 = max(0, min_col - pad_w)
    x2 = min(w - 1, max_col + pad_w)

    cropped = white_bg[y1:y2 + 1, x1:x2 + 1]
    cropped_pil = Image.fromarray(cropped)
    resized = cropped_pil.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    return resized, (int(x1), int(y1), int(x2), int(y2)), clothing_pct, clothing_px_count


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_valid_ids(csv_path, log):
    """Load CSV, filter by skip_reason and top-15 designers, return set of image_ids."""
    log.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Raw CSV: {len(df):,} rows")

    df = df[df["skip_reason"].isna()]
    log.info(f"After skip_reason filter: {len(df):,}")

    df = df[df["designer"].isin(TOP_15_DESIGNERS)]
    log.info(f"After top-15 designer filter: {len(df):,}")

    valid_ids = set(df["image_id"].astype(str).tolist())
    log.info(f"Valid image_ids: {len(valid_ids):,}")
    return valid_ids


def append_summary_row(row, summary_csv):
    """Append a single row to the summary CSV (creates header on first row)."""
    df = pd.DataFrame([row])
    if os.path.exists(summary_csv):
        df.to_csv(summary_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_csv, index=False)


def save_checkpoint(checkpoint_path, last_index, total_saved, total_skipped):
    with open(checkpoint_path, "w") as f:
        json.dump({
            "last_processed_index": int(last_index),
            "total_saved": int(total_saved),
            "total_skipped": int(total_skipped),
        }, f)


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0, 0, 0
    try:
        with open(checkpoint_path) as f:
            ck = json.load(f)
        return (int(ck.get("last_processed_index", 0)),
                int(ck.get("total_saved", 0)),
                int(ck.get("total_skipped", 0)))
    except Exception:
        return 0, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract clothing crops from Vogue runway dataset",
    )
    parser.add_argument("--csv_path", type=str,
                        default="/home/morayo/copalette/output/by_year/copalette_ALL_YEARS.csv")
    parser.add_argument("--output_dir", type=str,
                        default="/home/morayo/copalette/clothing/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    args.csv_path = os.path.abspath(args.csv_path)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Internal outputs (logs, checkpoint, summary CSV) live in script's outputs/
    script_outputs = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "outputs"),
    )
    os.makedirs(script_outputs, exist_ok=True)

    log = setup_logging(script_outputs)
    log.info("=" * 80)
    log.info("CLOTHING CROP EXTRACTION")
    log.info("=" * 80)
    log.info(f"CSV: {args.csv_path}")
    log.info(f"Output dir: {args.output_dir}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Resume: {not args.no_resume}")

    # Load valid IDs
    valid_ids = load_valid_ids(args.csv_path, log)

    # Load HuggingFace dataset
    log.info(f"Loading HF dataset: {DATASET_NAME}")
    from datasets import load_dataset
    hf_ds = load_dataset(DATASET_NAME, split="train")
    log.info(f"HF dataset: {len(hf_ds):,} images")

    # Load label_names once at startup
    label_names = hf_ds.features["label"].names
    log.info(f"Loaded {len(label_names)} label names")

    # Load model + processor
    log.info(f"Loading SegFormer model: {MODEL_NAME}")
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    model.to(device)
    model.eval()

    # Resume
    checkpoint_path = os.path.join(script_outputs, "checkpoint.json")
    summary_csv = os.path.join(script_outputs, "clothing_crop_summary.csv")

    if args.no_resume:
        start_index = 0
        total_saved = 0
        total_skipped = 0
        log.info("--no_resume: starting from index 0")
    else:
        start_index, total_saved, total_skipped = load_checkpoint(checkpoint_path)
        if start_index > 0:
            log.info(f"Resuming from index {start_index} "
                     f"(saved={total_saved}, skipped={total_skipped})")

    # Build dataset and loader from start_index onwards
    full_dataset = HFImageDataset(hf_ds, valid_ids, processor, label_names)

    # We use a sequential subset starting at start_index
    indices_to_process = list(range(start_index, len(hf_ds)))
    if not indices_to_process:
        log.info("Nothing to process. Exiting.")
        return

    log.info(f"Will process {len(indices_to_process):,} images "
             f"(from index {start_index} to {len(hf_ds) - 1})")

    # We can't easily use DataLoader with arbitrary sequential index lists AND
    # multiprocessing without subset semantics. Use torch Subset.
    from torch.utils.data import Subset
    subset = Subset(full_dataset, indices_to_process)
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn,
    )

    t_start = time.time()
    last_log_time = t_start
    n_processed_session = 0

    for batch_payload, _ in loader:
        if batch_payload is None:
            continue
        batch_indices, batch_ids, batch_images, pixel_values, skipped_meta = batch_payload

        # Log skipped (filtered) items quickly
        for (idx, iid, _img) in skipped_meta:
            n_processed_session += 1
            total_skipped += 1
            if iid is None:
                continue  # parse failure / not in valid set
            row = {
                "image_id": iid,
                "output_path": "",
                "clothing_px_count": 0,
                "clothing_pct_of_image": 0.0,
                "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
                "skip_reason": "filtered_or_parse_error",
            }
            try:
                append_summary_row(row, summary_csv)
            except Exception:
                pass

        # If no valid items in this batch, skip the model call entirely
        if pixel_values is None or len(batch_indices) == 0:
            continue

        # Run model on the batch
        try:
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # (B, num_labels, H', W')
            preds = logits.argmax(dim=1).cpu().numpy()  # (B, H', W')
        except Exception as e:
            log.error(f"Inference failed for batch: {e}")
            log.error(traceback.format_exc())
            # Mark whole batch as skipped
            for idx, iid in zip(batch_indices, batch_ids):
                n_processed_session += 1
                total_skipped += 1
                row = {
                    "image_id": iid,
                    "output_path": "",
                    "clothing_px_count": 0,
                    "clothing_pct_of_image": 0.0,
                    "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
                    "skip_reason": "inference_error",
                }
                try:
                    append_summary_row(row, summary_csv)
                except Exception:
                    pass
            continue

        # Process each item in the batch
        for i, (idx, image_id, pil_image) in enumerate(zip(batch_indices, batch_ids, batch_images)):
            n_processed_session += 1
            try:
                output_path = os.path.join(args.output_dir, f"{image_id}_clothing.jpg")

                # Skip if file already exists (and not forcing)
                if not args.no_resume and os.path.exists(output_path):
                    total_saved += 1  # treat as already-saved
                    continue

                # Build clothing mask at model output resolution
                pred_mask = preds[i]  # (H', W')
                clothing_mask = (pred_mask == CLOTHING_LABEL)

                result, bbox, pct, px_count = extract_clothing_crop(pil_image, clothing_mask)

                if result is None:
                    total_skipped += 1
                    row = {
                        "image_id": image_id,
                        "output_path": "",
                        "clothing_px_count": int(px_count),
                        "clothing_pct_of_image": 0.0,
                        "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
                        "skip_reason": "insufficient_clothing",
                    }
                    append_summary_row(row, summary_csv)
                    continue

                # Save JPEG
                result.save(output_path, "JPEG", quality=JPEG_QUALITY)
                total_saved += 1

                row = {
                    "image_id": image_id,
                    "output_path": output_path,
                    "clothing_px_count": int(px_count),
                    "clothing_pct_of_image": float(pct),
                    "bbox_x1": bbox[0], "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2], "bbox_y2": bbox[3],
                    "skip_reason": "",
                }
                append_summary_row(row, summary_csv)

            except Exception as e:
                total_skipped += 1
                log.error(f"Failed to process {image_id}: {e}")
                row = {
                    "image_id": image_id or f"index_{idx}",
                    "output_path": "",
                    "clothing_px_count": 0,
                    "clothing_pct_of_image": 0.0,
                    "bbox_x1": -1, "bbox_y1": -1, "bbox_x2": -1, "bbox_y2": -1,
                    "skip_reason": f"exception: {type(e).__name__}",
                }
                try:
                    append_summary_row(row, summary_csv)
                except Exception:
                    pass

        # Determine the maximum index processed in this batch for checkpointing
        max_idx = max(batch_indices) if batch_indices else (start_index + n_processed_session - 1)

        # Periodic logging + checkpoint
        if n_processed_session % CHECKPOINT_INTERVAL < args.batch_size:
            elapsed = time.time() - t_start
            rate = n_processed_session / max(elapsed, 1e-6)
            remaining = max(0, len(indices_to_process) - n_processed_session)
            eta_min = remaining / max(rate, 1e-6) / 60.0
            log.info(f"  Processed {n_processed_session:,}/{len(indices_to_process):,} "
                     f"| saved={total_saved:,} skipped={total_skipped:,} "
                     f"| {rate:.1f} img/s | ETA {eta_min:.1f} min")
            save_checkpoint(checkpoint_path, max_idx + 1, total_saved, total_skipped)

    # Final checkpoint
    save_checkpoint(checkpoint_path, len(hf_ds), total_saved, total_skipped)

    total_time = (time.time() - t_start) / 60.0
    log.info("=" * 80)
    log.info(f"DONE in {total_time:.1f} min")
    log.info(f"Processed (this session): {n_processed_session:,}")
    log.info(f"Total saved: {total_saved:,}")
    log.info(f"Total skipped: {total_skipped:,}")
    log.info(f"Output dir: {args.output_dir}")
    log.info(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
