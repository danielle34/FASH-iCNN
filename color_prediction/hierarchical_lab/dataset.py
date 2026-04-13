"""
Dataset for copalette_hierarchical_lab.

Replicates the EXACT filtering pipeline used by copalette_multimodal_color
so that the upstream face/clothing/designer F checkpoint loads cleanly:

  1. Drop skip_reason rows
  2. Require both face crop AND clothing crop on disk
  3. Compute c1 LAB, c1 CSS, c1 BK family
  4. Drop c1 in {black, gray}
  5. Drop palette black/gray weight > 40%
  6. Sort unique designers and assign 0..N-1 (15 expected)
  7. Map BK family -> chromatic 9-class index using the upstream order

Returns a dataframe whose `designer_id` and `bk_label` columns are aligned
with the F checkpoint's embedding and head respectively.

The dataset class is multimodal: returns
    (face_img, clothing_img, designer_id, true_bk_idx, true_lab, image_id, true_css)
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
    css_name_to_berlin_kay, compute_blackgray_weight,
    CHROMATIC_BK_TO_IDX, CHROMATIC_BK_NAMES,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Loading and filtering (matches multimodal_color exactly)
# ---------------------------------------------------------------------------
def load_and_filter(csv_path, face_crops_dir, clothing_crops_dir, output_dir):
    cache_path = os.path.join(output_dir, "hierarchical_lab_dataset.csv")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached dataset from {cache_path}")
        df = pd.read_csv(cache_path)
        designer_names = sorted(df["designer"].unique().tolist())
        designer_to_idx = {d: i for i, d in enumerate(designer_names)}
        logger.info(f"Cached dataset: {len(df)} rows, {len(designer_to_idx)} designers")
        return df, designer_to_idx

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

    df = df[df["image_id"].apply(both_exist)].copy()
    logger.info(f"After both-crops filter: {len(df)} rows")

    # Skin LAB
    skin_rgb = df[["skin_r_viz", "skin_g_viz", "skin_b_viz"]].values.astype(np.float64)
    skin_lab = rgb_to_lab_array(skin_rgb)
    df["skin_L"] = skin_lab[:, 0]
    df["skin_a"] = skin_lab[:, 1]
    df["skin_b_lab"] = skin_lab[:, 2]

    # c1 LAB
    c1_rgb = df[["c1_r", "c1_g", "c1_b"]].values.astype(np.float64)
    c1_lab = rgb_to_lab_array(c1_rgb)
    df["c1_L"] = c1_lab[:, 0]
    df["c1_a"] = c1_lab[:, 1]
    df["c1_b_lab"] = c1_lab[:, 2]

    # CSS + BK derived from c1 LAB
    df["c1_css_name"] = lab_to_css_name(c1_lab)
    df["c1_berlin_kay"] = [css_name_to_berlin_kay(n) for n in df["c1_css_name"]]

    # Filter c1 black/gray
    n_before = len(df)
    df = df[~df["c1_berlin_kay"].isin(["black", "gray"])].reset_index(drop=True)
    logger.info(f"Drop c1 black/gray: {n_before} -> {len(df)} rows")

    # Filter palette black/gray weight > 40%
    n_before = len(df)
    bg = df.apply(compute_blackgray_weight, axis=1)
    df = df[bg <= 0.40].reset_index(drop=True)
    logger.info(f"Drop palette > 40%% black/gray: {n_before} -> {len(df)} rows")

    # Designer encoding (sorted unique — matches multimodal_color)
    designer_names = sorted(df["designer"].unique().tolist())
    designer_to_idx = {d: i for i, d in enumerate(designer_names)}
    df["designer_id"] = df["designer"].map(designer_to_idx).astype(int)

    # BK 9-class label using the chromatic ordering
    df["bk_label"] = df["c1_berlin_kay"].map(CHROMATIC_BK_TO_IDX).astype(int)

    logger.info(f"Final: {len(df)} rows, {len(designer_to_idx)} designers")
    logger.info(f"BK distribution:")
    for n, c in df["c1_berlin_kay"].value_counts().items():
        logger.info(f"  {n}: {c}")

    df.to_csv(cache_path, index=False)
    logger.info(f"Cached dataset to {cache_path}")
    return df, designer_to_idx


# ---------------------------------------------------------------------------
# Stratified split (by BK family — matches multimodal_color)
# ---------------------------------------------------------------------------
def stratified_split(df, seed=42, strat_col="c1_berlin_kay"):
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby(strat_col):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_parts.append(df.loc[idx[:n_train]])
        val_parts.append(df.loc[idx[n_train:n_train + n_val]])
        test_parts.append(df.loc[idx[n_train + n_val:]])
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    logger.info(f"Stratified split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class HierarchicalLABDataset(Dataset):
    """
    Returns:
        face_img         (3, 224, 224)
        clothing_img     (3, 224, 224)
        designer_id      int
        bk_label         int  (chromatic 9-class)
        true_lab         (3,) float32
        image_id         str
        true_css         str
    """

    def __init__(self, df, face_crops_dir, clothing_crops_dir, transform):
        self.df = df.reset_index(drop=True)
        self.face_crops_dir = face_crops_dir
        self.clothing_crops_dir = clothing_crops_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        face_path = os.path.join(self.face_crops_dir, f"{image_id}_face.jpg")
        cloth_path = os.path.join(self.clothing_crops_dir, f"{image_id}_clothing.jpg")
        face = Image.open(face_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")
        face_t = self.transform(face)
        cloth_t = self.transform(cloth)

        designer_id = int(row["designer_id"])
        bk_label = int(row["bk_label"])
        true_lab = torch.tensor(
            [row["c1_L"], row["c1_a"], row["c1_b_lab"]], dtype=torch.float32,
        )
        true_css = str(row["c1_css_name"])

        return face_t, cloth_t, designer_id, bk_label, true_lab, str(image_id), true_css


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
NUM_WORKERS = 4


def make_loader(df, face_crops_dir, clothing_crops_dir, transform,
                batch_size=64, shuffle=True):
    ds = HierarchicalLABDataset(df, face_crops_dir, clothing_crops_dir, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)
