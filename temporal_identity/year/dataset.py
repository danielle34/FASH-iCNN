"""
Dataset for clothing-only year prediction (1991-2024, 34-class).

Filtering: skip_reason null, clothing crop on disk, year ∈ [1991, 2024].
No chromatic / black-gray filter — year prediction uses the full corpus.
Both random stratified 70/15/15 and temporal (train ≤2013 / val 2014-2016
/ test ≥2017) splits are supported.
Self-contained — no imports from other copalette modules.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

log = logging.getLogger("clothing_year")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2

YEAR_MIN = 1991
YEAR_MAX = 2024
NUM_YEARS = YEAR_MAX - YEAR_MIN + 1  # 34

DECADE_BINS = [
    (1991, 2000, "1991-2000"),
    (2001, 2010, "2001-2010"),
    (2011, 2020, "2011-2020"),
    (2021, 2024, "2021-2024"),
]
DECADE_LABELS = [d[2] for d in DECADE_BINS]
DECADE_TO_IDX = {label: i for i, label in enumerate(DECADE_LABELS)}
NUM_DECADES = len(DECADE_BINS)


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


def assign_decade(year):
    for lo, hi, label in DECADE_BINS:
        if lo <= year <= hi:
            return label
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path, clothing_crops_dir, output_dir):
    cache_path = os.path.join(output_dir, "dataset.csv")
    if os.path.exists(cache_path):
        log.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, low_memory=False)
        log.info(f"Loaded {len(df):,} rows from cache")
        return df

    clothing_dir = Path(clothing_crops_dir)
    raw = pd.read_csv(csv_path, low_memory=False)
    log.info(f"Raw CSV: {len(raw):,} rows")

    df = raw[raw["skip_reason"].isna()].copy()
    log.info(f"After skip_reason filter: {len(df):,}")

    df["_clothing_exists"] = df["image_id"].apply(
        lambda x: (clothing_dir / f"{x}_clothing.jpg").exists()
    )
    log.info(f"Clothing crops found: {df['_clothing_exists'].sum():,}")
    df = df[df["_clothing_exists"]].drop(columns=["_clothing_exists"]).reset_index(drop=True)
    log.info(f"After clothing-crop filter: {len(df):,}")

    # Drop rows without a parseable year
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].reset_index(drop=True)
    log.info(f"After year-range filter ({YEAR_MIN}-{YEAR_MAX}): {len(df):,}")

    # Year label index
    df["year_label"] = (df["year"] - YEAR_MIN).astype(int)

    # Decade label
    df["decade"] = df["year"].apply(assign_decade)
    df = df[df["decade"].notna()].reset_index(drop=True)
    df["decade_label"] = df["decade"].map(DECADE_TO_IDX)

    log.info(f"Final: {len(df):,} rows | year range {df['year'].min()}-{df['year'].max()}")
    log.info(f"Year distribution (top 10):\n"
             f"{df['year'].value_counts().sort_index().tail(10).to_string()}")
    log.info(f"Decade distribution:\n{df['decade'].value_counts().to_string()}")

    df.to_csv(cache_path, index=False)
    log.info(f"Saved preprocessed dataset to {cache_path}")
    return df


def stratified_split_70_15_15(df, strat_col="year", seed=42):
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


def split_temporal(df):
    """Train: <=2013, Val: 2014-2016, Test: 2017-2024."""
    train = df[df["year"] <= 2013].copy()
    val = df[(df["year"] >= 2014) & (df["year"] <= 2016)].copy()
    test = df[df["year"] >= 2017].copy()
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ClothingYearDataset(Dataset):
    """Returns (img, label) — label is either year_label (0..33) or decade_label (0..3)."""

    def __init__(self, dataframe, clothing_crops_dir, transform, label_column):
        self.df = dataframe.reset_index(drop=True)
        self.clothing_dir = Path(clothing_crops_dir)
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.clothing_dir / f"{row['image_id']}_clothing.jpg"
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return img, label


def make_loader(df, clothing_dir, transform, label_col, batch_size, shuffle):
    ds = ClothingYearDataset(df, clothing_dir, transform, label_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
