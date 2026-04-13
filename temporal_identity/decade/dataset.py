"""
Dataset for copalette_clothing_decade.

Loads rows where BOTH face crop and clothing crop exist on disk, assigns a
decade label from the year column, and provides a Dataset that returns
either the clothing crop or the face crop depending on `mode`.

Decades:
    1991-2000 -> 0
    2001-2010 -> 1
    2011-2020 -> 2
    2021-2024 -> 3
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Decade mapping
# ---------------------------------------------------------------------------
DECADES = [
    ("1991-2000", 1991, 2000),
    ("2001-2010", 2001, 2010),
    ("2011-2020", 2011, 2020),
    ("2021-2024", 2021, 2024),
]
DECADE_NAMES = [d[0] for d in DECADES]
NUM_DECADES = len(DECADES)


def year_to_decade_idx(year):
    """Return decade index 0..3 or None if out of range / invalid."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    for i, (_, lo, hi) in enumerate(DECADES):
        if lo <= y <= hi:
            return i
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
def load_and_filter(csv_path, face_crops_dir, clothing_crops_dir, output_dir):
    """
    Drop skip_reason rows, require both crops on disk, assign decade,
    drop rows with year outside 1991-2024. Cache as outputs/dataset.csv.
    """
    cache_path = os.path.join(output_dir, "dataset.csv")
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

    def both_exist(image_id):
        f = os.path.join(face_crops_dir, f"{image_id}_face.jpg")
        c = os.path.join(clothing_crops_dir, f"{image_id}_clothing.jpg")
        return os.path.isfile(f) and os.path.isfile(c)

    mask = df["image_id"].apply(both_exist)
    df = df[mask].copy()
    logger.info(f"After face+clothing crop check: {len(df)} rows")

    # Decade assignment
    df["decade_idx"] = df["year"].apply(year_to_decade_idx)
    n_before = len(df)
    df = df[df["decade_idx"].notna()].copy()
    df["decade_idx"] = df["decade_idx"].astype(int)
    logger.info(f"Drop rows with year outside 1991-2024: {n_before} -> {len(df)} rows")

    logger.info("Decade distribution:")
    for i, name in enumerate(DECADE_NAMES):
        cnt = (df["decade_idx"] == i).sum()
        logger.info(f"  {name}: {cnt} ({cnt/len(df)*100:.1f}%)")

    df.to_csv(cache_path, index=False)
    logger.info(f"Cached dataset to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(df, seed=42):
    """Random 70/15/15 stratified by decade."""
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for d_idx in range(NUM_DECADES):
        sub = df[df["decade_idx"] == d_idx]
        if len(sub) == 0:
            continue
        idx = rng.permutation(len(sub))
        n_train = int(0.70 * len(sub))
        n_val = int(0.15 * len(sub))
        train_parts.append(sub.iloc[idx[:n_train]])
        val_parts.append(sub.iloc[idx[n_train:n_train + n_val]])
        test_parts.append(sub.iloc[idx[n_train + n_val:]])
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    logger.info(f"Stratified split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class DecadeDataset(Dataset):
    """
    Returns (image_tensor, label).

    mode controls which crop is loaded:
        "clothing" -> {clothing_crops_dir}/{image_id}_clothing.jpg
        "face"     -> {face_crops_dir}/{image_id}_face.jpg
    """

    def __init__(self, df, face_crops_dir, clothing_crops_dir,
                 transform, mode):
        assert mode in ("clothing", "face"), f"Unknown mode: {mode}"
        self.df = df.reset_index(drop=True)
        self.face_crops_dir = face_crops_dir
        self.clothing_crops_dir = clothing_crops_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        if self.mode == "clothing":
            path = os.path.join(self.clothing_crops_dir, f"{image_id}_clothing.jpg")
        else:
            path = os.path.join(self.face_crops_dir, f"{image_id}_face.jpg")
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        label = int(row["decade_idx"])
        return img_tensor, label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
NUM_WORKERS = 8


def make_loader(df, face_crops_dir, clothing_crops_dir, transform, mode,
                batch_size=128, shuffle=True):
    ds = DecadeDataset(df, face_crops_dir, clothing_crops_dir, transform, mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)
