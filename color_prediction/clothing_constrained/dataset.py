"""
Dataset for copalette_clothing_constrained.

Loads rows where the clothing crop ({image_id}_clothing.jpg) exists,
applies the copalette_noblack chromatic filter, restricts to the top-15
designers (by clothing-crop count after filtering, with a minimum of 200
chromatic images per designer), and assigns decade labels.

Provides a single Dataset class that returns either BK or CSS labels via
the `label_column` parameter.
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from colors import (
    rgb_to_lab_array, lab_to_berlin_kay, lab_to_css_name,
    compute_blackgray_weight, CHROMATIC_BK_NAMES, BLACKGRAY_CSS,
    CSS_LAB_ARRAY, CSS_NAMES,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Decade definitions (matches copalette_clothing_decade)
DECADES = [
    ("1991-2000", 1991, 2000),
    ("2001-2010", 2001, 2010),
    ("2011-2020", 2011, 2020),
    ("2021-2024", 2021, 2024),
]
DECADE_NAMES = [d[0] for d in DECADES]


def year_to_decade(year):
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    for name, lo, hi in DECADES:
        if lo <= y <= hi:
            return name
    return None


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------
def load_and_filter(csv_path, clothing_crops_dir, output_dir,
                    num_designers=15, min_designer_count=200):
    """
    Drop skip_reason rows, require clothing crop on disk, derive c1
    LAB/BK/CSS, apply chromatic black/gray filter, keep only the top-N
    designers (with >= min_designer_count chromatic images each), assign
    decades. Cache as outputs/clothing_constrained_dataset.csv.
    """
    cache_path = os.path.join(output_dir, "clothing_constrained_dataset.csv")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        logger.info(f"Cached dataset: {len(df)} rows")
        return df

    logger.info(f"Loading raw CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Raw dataset: {len(df)} rows")

    if "skip_reason" in df.columns:
        df = df[df["skip_reason"].isna()].copy()
        logger.info(f"After skip_reason filter: {len(df)} rows")

    def cloth_exists(image_id):
        return os.path.isfile(
            os.path.join(clothing_crops_dir, f"{image_id}_clothing.jpg")
        )

    df = df[df["image_id"].apply(cloth_exists)].copy()
    logger.info(f"After clothing crop check: {len(df)} rows")

    # c1 LAB / BK / CSS
    c1_rgb = df[["c1_r", "c1_g", "c1_b"]].values.astype(np.float64)
    c1_lab = rgb_to_lab_array(c1_rgb)
    df["c1_L"] = c1_lab[:, 0]
    df["c1_a"] = c1_lab[:, 1]
    df["c1_b_lab"] = c1_lab[:, 2]
    df["c1_bk"] = lab_to_berlin_kay(c1_lab)
    df["c1_css"] = lab_to_css_name(c1_lab)

    # Chromatic black/gray filter
    n_before = len(df)
    df = df[~df["c1_bk"].isin(["black", "gray"])].copy()
    logger.info(f"Filter c1 black/gray: {n_before} -> {len(df)} rows")

    n_before = len(df)
    df["_bg_weight"] = df.apply(compute_blackgray_weight, axis=1)
    df = df[df["_bg_weight"] <= 0.40].copy()
    logger.info(f"Filter palette > 40%% black/gray: {n_before} -> {len(df)} rows")

    # Top-N designers by chromatic count
    d_counts = df["designer"].value_counts()
    eligible = d_counts[d_counts >= min_designer_count].head(num_designers)
    keep = sorted(eligible.index.tolist())
    df = df[df["designer"].isin(keep)].copy()
    logger.info(f"Top {num_designers} designers (>= {min_designer_count}): {keep}")
    logger.info(f"After designer filter: {len(df)} rows")

    # Decade assignment
    df["decade"] = df["year"].apply(year_to_decade)
    df = df[df["decade"].notna()].copy()
    logger.info(f"After decade filter: {len(df)} rows")

    df = df.reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    logger.info(f"Cached dataset to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------
def build_bk_class_mapping():
    class_names = sorted(CHROMATIC_BK_NAMES)
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    return class_names, class_to_idx


def build_css_class_mapping(train_df, min_count=10):
    """CSS class mapping for one slice; rare classes remapped to nearest frequent."""
    counts = train_df["c1_css"].value_counts()
    chromatic = {n: c for n, c in counts.items() if n not in BLACKGRAY_CSS}
    frequent = {n for n, c in chromatic.items() if c >= min_count}
    rare = {n for n, c in chromatic.items() if c < min_count}

    freq_list = sorted(frequent)
    if not freq_list:
        freq_list = sorted(chromatic.keys())
        rare = set()

    freq_lab = np.array([CSS_LAB_ARRAY[CSS_NAMES.index(n)] for n in freq_list])

    remap = {}
    for name in rare:
        idx = CSS_NAMES.index(name)
        dists = np.linalg.norm(freq_lab - CSS_LAB_ARRAY[idx], axis=1)
        remap[name] = freq_list[int(np.argmin(dists))]
    for name in frequent:
        remap[name] = name
    for name in BLACKGRAY_CSS:
        idx = CSS_NAMES.index(name)
        dists = np.linalg.norm(freq_lab - CSS_LAB_ARRAY[idx], axis=1)
        remap[name] = freq_list[int(np.argmin(dists))]

    class_names = sorted(frequent)
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    return remap, class_names, class_to_idx


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def random_split(df, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_train = int(0.70 * len(df))
    n_val = int(0.15 * len(df))
    train = df.iloc[idx[:n_train]].reset_index(drop=True)
    val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class ClothingDataset(Dataset):
    """
    Returns (image_tensor, label).
    Loads only the clothing crop — no face crop is needed for any
    experiment in this folder.
    """

    def __init__(self, df, clothing_crops_dir, transform,
                 label_column, class_to_idx, remap=None):
        self.df = df.reset_index(drop=True)
        self.clothing_crops_dir = clothing_crops_dir
        self.transform = transform
        self.label_column = label_column
        self.class_to_idx = class_to_idx
        self.remap = remap or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        path = os.path.join(self.clothing_crops_dir, f"{image_id}_clothing.jpg")
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)

        raw_label = row[self.label_column]
        mapped = self.remap.get(raw_label, raw_label)
        label = self.class_to_idx.get(mapped, 0)
        return img_tensor, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
NUM_WORKERS = 8


def make_loader(df, clothing_crops_dir, transform,
                label_column, class_to_idx, remap=None,
                batch_size=64, shuffle=True):
    ds = ClothingDataset(df, clothing_crops_dir, transform,
                         label_column, class_to_idx, remap=remap)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)
