"""
Dataset for face + silhouette designer classification.

Filtering: skip_reason null, top-15 designers, BOTH face crop AND
silhouette on disk. **No** chromatic / black-gray filter — designer
prediction uses the full dataset. Stratified 70/15/15 split by designer.
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

log = logging.getLogger("silhouette_designer")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2

TOP_15_DESIGNERS = [
    "alexander mcqueen", "armani prive", "balenciaga",
    "calvin klein collection", "chanel", "christian dior",
    "fendi", "gucci", "hermes", "louis vuitton", "prada",
    "ralph lauren", "saint laurent", "valentino", "versace",
]


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


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path, face_crops_dir, silhouette_dir, output_dir):
    cache_path = os.path.join(output_dir, "dataset.csv")
    if os.path.exists(cache_path):
        log.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        designers_present = sorted(df["designer"].unique().tolist())
        designer_to_idx = {d: i for i, d in enumerate(designers_present)}
        log.info(f"Loaded {len(df):,} rows, {len(designer_to_idx)} designers from cache")
        return df, designer_to_idx

    face_dir = Path(face_crops_dir)
    sil_dir = Path(silhouette_dir)
    raw = pd.read_csv(csv_path)
    log.info(f"Raw CSV: {len(raw):,} rows")

    df = raw[raw["skip_reason"].isna()].copy()
    log.info(f"After skip_reason filter: {len(df):,}")

    df = df[df["designer"].isin(TOP_15_DESIGNERS)].reset_index(drop=True)
    log.info(f"After top-15 designer filter: {len(df):,}")

    df["_face_exists"] = df["image_id"].apply(
        lambda x: (face_dir / f"{x}_face.jpg").exists()
    )
    df["_sil_exists"] = df["image_id"].apply(
        lambda x: (sil_dir / f"{x}_silhouette.jpg").exists()
    )
    log.info(f"Face: {df['_face_exists'].sum():,}, "
             f"silhouette: {df['_sil_exists'].sum():,}")
    df = df[df["_face_exists"] & df["_sil_exists"]].drop(
        columns=["_face_exists", "_sil_exists"]
    ).reset_index(drop=True)
    log.info(f"After both-modalities filter: {len(df):,}")

    designers_present = sorted(df["designer"].unique().tolist())
    designer_to_idx = {d: i for i, d in enumerate(designers_present)}
    df["designer_label"] = df["designer"].map(designer_to_idx)

    log.info(f"Designers ({len(designer_to_idx)}): {designers_present}")
    log.info(f"Designer distribution:\n{df['designer'].value_counts().to_string()}")

    df.to_csv(cache_path, index=False)
    log.info(f"Saved preprocessed dataset to {cache_path}")
    return df, designer_to_idx


def stratified_split_70_15_15(df, strat_col="designer", seed=42):
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
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class FaceSilhouetteDataset(Dataset):
    """Returns (face_img, silhouette_img, designer_label).

    Disabled streams return zero tensors so the model can decide whether to
    consume them. The dataset uses (face_dir, sil_dir) and the suffixes
    `_face.jpg` / `_silhouette.jpg`.
    """

    def __init__(self, dataframe, face_crops_dir, silhouette_dir, transform,
                 use_face=True, use_silhouette=True):
        self.df = dataframe.reset_index(drop=True)
        self.face_dir = Path(face_crops_dir)
        self.sil_dir = Path(silhouette_dir)
        self.transform = transform
        self.use_face = use_face
        self.use_silhouette = use_silhouette

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.use_face:
            face_img = Image.open(self.face_dir / f"{row['image_id']}_face.jpg").convert("RGB")
            face_img = self.transform(face_img)
        else:
            face_img = torch.zeros(3, 224, 224, dtype=torch.float32)

        if self.use_silhouette:
            sil_img = Image.open(self.sil_dir / f"{row['image_id']}_silhouette.jpg").convert("RGB")
            sil_img = self.transform(sil_img)
        else:
            sil_img = torch.zeros(3, 224, 224, dtype=torch.float32)

        label = torch.tensor(int(row["designer_label"]), dtype=torch.long)
        return face_img, sil_img, label


def make_loader(df, face_dir, silhouette_dir, transform, batch_size, shuffle,
                use_face, use_silhouette):
    ds = FaceSilhouetteDataset(df, face_dir, silhouette_dir, transform,
                               use_face=use_face, use_silhouette=use_silhouette)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
