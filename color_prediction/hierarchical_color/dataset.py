"""
Dataset for hierarchical color experiments.

Same dual black/gray filter as no_black variant. Requires both face and
clothing crops on disk. Encodes c1 BK 9-class label and c1 CSS label.
Self-contained — no imports from other copalette modules.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from colors import (
    rgb_to_lab_array, lab_to_berlin_kay, lab_to_css_name,
    css_name_to_berlin_kay,
)

log = logging.getLogger("hierarchical")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2

CHROMATIC_BK_NAMES = ["white", "red", "orange", "yellow", "green",
                     "blue", "purple", "pink", "brown"]
CHROMATIC_BK_TO_IDX = {n: i for i, n in enumerate(CHROMATIC_BK_NAMES)}
CHROMATIC_BK_IDX_TO_NAME = {i: n for n, i in CHROMATIC_BK_TO_IDX.items()}


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


def _bg_palette_weight(row):
    w = 0.0
    for i in range(1, 7):
        bk = f"c{i}_berlin_kay"
        pct = f"c{i}_pct"
        if bk in row.index and pct in row.index:
            if row[bk] in ("black", "gray"):
                w += row[pct]
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path, face_crops_dir, clothing_crops_dir, output_dir):
    cache_path = os.path.join(output_dir, "hierarchical_dataset.csv")
    if os.path.exists(cache_path):
        log.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        designer_names = sorted(df["designer"].unique().tolist())
        designer_to_idx = {d: i for i, d in enumerate(designer_names)}
        log.info(f"Loaded {len(df):,} rows, {len(designer_to_idx)} designers from cache")
        return df, designer_to_idx

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

    # Designer encoding
    designer_names = sorted(df["designer"].unique().tolist())
    designer_to_idx = {d: i for i, d in enumerate(designer_names)}
    df["designer_id"] = df["designer"].map(designer_to_idx)

    # BK 9-class label
    df["bk_label"] = df["c1_berlin_kay"].map(CHROMATIC_BK_TO_IDX)

    log.info(f"Final: {len(df):,} rows, {len(designer_to_idx)} designers")
    log.info(f"BK distribution:\n{df['c1_berlin_kay'].value_counts().to_string()}")

    df.to_csv(cache_path, index=False)
    log.info(f"Saved preprocessed dataset to {cache_path}")
    return df, designer_to_idx


def stratified_split_70_15_15(df, strat_col="c1_berlin_kay", seed=42):
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class MultimodalDataset(Dataset):
    """Returns (face, clothing, designer_id, bk_label) — used for Stage 1 inference."""

    def __init__(self, dataframe, face_crops_dir, clothing_crops_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.face_dir = Path(face_crops_dir)
        self.clothing_dir = Path(clothing_crops_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        face_img = Image.open(self.face_dir / f"{row['image_id']}_face.jpg").convert("RGB")
        face_img = self.transform(face_img)
        cloth_img = Image.open(self.clothing_dir / f"{row['image_id']}_clothing.jpg").convert("RGB")
        cloth_img = self.transform(cloth_img)
        designer_id = torch.tensor(int(row["designer_id"]), dtype=torch.long)
        label = torch.tensor(int(row["bk_label"]), dtype=torch.long)
        return face_img, cloth_img, designer_id, label


class FamilyCSSDataset(Dataset):
    """Returns (face, clothing, css_local_label) for Stage 2 family models."""

    def __init__(self, dataframe, face_crops_dir, clothing_crops_dir,
                 transform, label_column):
        self.df = dataframe.reset_index(drop=True)
        self.face_dir = Path(face_crops_dir)
        self.clothing_dir = Path(clothing_crops_dir)
        self.transform = transform
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        face_img = Image.open(self.face_dir / f"{row['image_id']}_face.jpg").convert("RGB")
        face_img = self.transform(face_img)
        cloth_img = Image.open(self.clothing_dir / f"{row['image_id']}_clothing.jpg").convert("RGB")
        cloth_img = self.transform(cloth_img)
        label = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return face_img, cloth_img, label


def make_multimodal_loader(df, face_dir, clothing_dir, transform, batch_size, shuffle):
    ds = MultimodalDataset(df, face_dir, clothing_dir, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)


def make_family_loader(df, face_dir, clothing_dir, transform, label_col,
                       batch_size, shuffle):
    ds = FamilyCSSDataset(df, face_dir, clothing_dir, transform, label_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
