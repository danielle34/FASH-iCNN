"""
Dataset for face / clothing CSS named-color experiments.

Same dual black/gray filter as no_black variant. Both face and clothing
crops required on disk. CSS labels built from training rows where the
c1 CSS class has >= 30 training examples; classes below that are dropped.
Stratified 70/15/15 split by `c1_css_name`.

Supports two modes per condition:
  - regular dual-stream  : uses face + clothing (or either alone), color
  - grayscale clothing   : converts the clothing crop to grayscale before
                           ImageNet normalization, replicating the
                           "structure-only" condition

Self-contained — no imports from other copalette modules.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from colors import (
    rgb_to_lab_array, lab_to_berlin_kay, lab_to_css_name,
    css_name_to_berlin_kay,
)

log = logging.getLogger("copalette_css_clothing")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2

MIN_TRAIN_PER_CLASS = 30


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Grayscale variants — convert to 3-channel grayscale BEFORE normalization
# so the EfficientNet's expected input shape stays (3, 224, 224).
gray_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

gray_eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _bg_palette_weight(row):
    w = 0.0
    for i in range(1, 7):
        bk = f"c{i}_berlin_kay"
        pct = f"c{i}_pct"
        if bk in row.index and pct in row.index:
            if row[bk] in ("black", "gray"):
                w += row[pct]
    return w


def load_and_preprocess(csv_path, face_crops_dir, clothing_crops_dir, output_dir):
    cache_path = os.path.join(output_dir, "dataset.csv")
    if os.path.exists(cache_path):
        log.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        log.info(f"Loaded {len(df):,} rows from cache")
        return df

    face_dir = Path(face_crops_dir)
    clothing_dir = Path(clothing_crops_dir)
    raw = pd.read_csv(csv_path)
    log.info(f"Raw CSV: {len(raw):,} rows")

    df = raw[raw["skip_reason"].isna()].copy()
    log.info(f"After skip_reason filter: {len(df):,}")

    df["_face_exists"] = df["image_id"].apply(
        lambda x: (face_dir / f"{x}_face.jpg").exists()
    )
    df["_clothing_exists"] = df["image_id"].apply(
        lambda x: (clothing_dir / f"{x}_clothing.jpg").exists()
    )
    log.info(f"Face: {df['_face_exists'].sum():,}, "
             f"clothing: {df['_clothing_exists'].sum():,}")
    df = df[df["_face_exists"] & df["_clothing_exists"]].drop(
        columns=["_face_exists", "_clothing_exists"]
    ).reset_index(drop=True)
    log.info(f"After both-crops filter: {len(df):,}")

    # All 6 clothing colors -> LAB + BK
    for i in range(1, 7):
        ci_rgb = df[[f"c{i}_r", f"c{i}_g", f"c{i}_b"]].values.astype(np.float64)
        ci_lab = rgb_to_lab_array(ci_rgb)
        df[f"c{i}_L"] = ci_lab[:, 0]
        df[f"c{i}_a"] = ci_lab[:, 1]
        df[f"c{i}_b_lab"] = ci_lab[:, 2]
        df[f"c{i}_berlin_kay"] = lab_to_berlin_kay(ci_lab)

    # CSS + BK for c1
    c1_lab = df[["c1_L", "c1_a", "c1_b_lab"]].values
    df["c1_css_name"] = lab_to_css_name(c1_lab)
    df["c1_berlin_kay"] = [css_name_to_berlin_kay(n) for n in df["c1_css_name"]]

    # Filter c1 black/gray
    n_before = len(df)
    df = df[~df["c1_berlin_kay"].isin(["black", "gray"])].reset_index(drop=True)
    log.info(f"Dropped {n_before - len(df):,} c1 black/gray -> {len(df):,}")

    # Filter palette black/gray weight > 40%
    n_before = len(df)
    bg = df.apply(_bg_palette_weight, axis=1)
    df = df[bg <= 0.40].reset_index(drop=True)
    log.info(f"Dropped {n_before - len(df):,} dominant black/gray -> {len(df):,}")

    log.info(f"Final: {len(df):,} chromatic rows")
    log.info(f"CSS top-20:\n{df['c1_css_name'].value_counts().head(20).to_string()}")

    df.to_csv(cache_path, index=False)
    log.info(f"Saved preprocessed dataset to {cache_path}")
    return df


def stratified_split_70_15_15(df, strat_col="c1_css_name", seed=42):
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby(strat_col):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_parts.append(group.loc[idx[:n_train]])
        val_parts.append(group.loc[idx[n_train:n_train + n_val]])
        test_parts.append(group.loc[idx[n_train + n_val:]])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val, test


def build_css_label_mapping(train_df, min_count=MIN_TRAIN_PER_CLASS):
    """Build CSS-name -> local index using only training rows with >= min_count."""
    counts = train_df["c1_css_name"].value_counts()
    valid = counts[counts >= min_count].index.tolist()
    name_to_idx = {c: i for i, c in enumerate(sorted(valid))}
    return name_to_idx


def filter_to_valid_css(df, name_to_idx):
    """Filter rows to those whose c1_css_name is in name_to_idx, add css_label column."""
    out = df[df["c1_css_name"].isin(name_to_idx)].copy()
    out["css_label"] = out["c1_css_name"].map(name_to_idx)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DualCropDataset(Dataset):
    """Returns (face_img, clothing_img, label).

    Either stream can be disabled (returns zeros) to support face-only or
    clothing-only ablations. The clothing stream can also be forced to
    grayscale by passing a grayscale transform.
    """

    def __init__(self, dataframe, face_crops_dir, clothing_crops_dir,
                 face_transform, clothing_transform, label_column,
                 use_face=True, use_clothing=True):
        self.df = dataframe.reset_index(drop=True)
        self.face_dir = Path(face_crops_dir)
        self.clothing_dir = Path(clothing_crops_dir)
        self.face_transform = face_transform
        self.clothing_transform = clothing_transform
        self.label_column = label_column
        self.use_face = use_face
        self.use_clothing = use_clothing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.use_face:
            face_img = Image.open(self.face_dir / f"{row['image_id']}_face.jpg").convert("RGB")
            face_img = self.face_transform(face_img)
        else:
            face_img = torch.zeros(3, 224, 224, dtype=torch.float32)

        if self.use_clothing:
            cloth_img = Image.open(self.clothing_dir / f"{row['image_id']}_clothing.jpg").convert("RGB")
            cloth_img = self.clothing_transform(cloth_img)
        else:
            cloth_img = torch.zeros(3, 224, 224, dtype=torch.float32)

        label = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return face_img, cloth_img, label


def make_loader(df, face_dir, clothing_dir, face_transform, clothing_transform,
                label_col, batch_size, shuffle, use_face, use_clothing):
    ds = DualCropDataset(df, face_dir, clothing_dir, face_transform, clothing_transform,
                         label_col, use_face=use_face, use_clothing=use_clothing)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
