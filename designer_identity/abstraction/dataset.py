"""
Dataset for the clothing-crop abstraction-ladder designer experiment.

Four conditions, all training a single-stream EfficientNet-B0 over the
same 15-way designer target but with four progressively abstracted views
of the clothing crop:

  1. full color  — clothing crop as-is, RGB
  2. grayscale   — clothing crop desaturated, replicated to 3 channels
  3. silhouette  — preprocessed silhouette from copalette_silhouette_designer
                    if available, otherwise filled binary mask generated
                    from the clothing crop on the fly
  4. edge        — preprocessed edge map from /home/morayo/copalette/edges/
                    if available, otherwise Canny edges generated on the
                    fly from the clothing crop

All four variants emit a 3-channel, ImageNet-normalized 224x224 tensor so
the same training loop drives every condition without branching.

Self-contained — no imports from other copalette modules.
"""

import os
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

log = logging.getLogger("abstraction_designer")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_WORKERS = 2

TOP_15_DESIGNERS = [
    "alexander mcqueen", "armani prive", "balenciaga",
    "calvin klein collection", "chanel", "christian dior",
    "fendi", "gucci", "hermes", "louis vuitton", "prada",
    "ralph lauren", "saint laurent", "valentino", "versace",
]

CONDITIONS = ["fullcolor", "grayscale", "silhouette", "edge"]

# Default paths for pre-generated silhouettes and edges
DEFAULT_SILHOUETTE_DIR = "/home/morayo/copalette/copalette_silhouette_designer/outputs/"
DEFAULT_EDGE_DIR = "/home/morayo/copalette/edges/"


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

# Used for training augmentation on the RGB branches (full color + grayscale)
train_transform_color = transforms.Compose([
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

# For grayscale / silhouette / edge we skip ColorJitter (no color to jitter)
# and do a softer augmentation that preserves the abstracted signal.
train_transform_mono = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path, clothing_crops_dir, output_dir):
    """Build the filtered master dataframe.

    Drops skip_reason rows, restricts to top-15 designers, requires the
    clothing crop to exist on disk. Does NOT require silhouette or edge
    files — those are generated on the fly when missing so this filter
    stays stable across all four conditions.
    """
    cache_path = os.path.join(output_dir, "dataset.csv")
    if os.path.exists(cache_path):
        log.info(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
        designers_present = sorted(df["designer"].unique().tolist())
        designer_to_idx = {d: i for i, d in enumerate(designers_present)}
        log.info(f"Loaded {len(df):,} rows, {len(designer_to_idx)} designers")
        return df, designer_to_idx

    clothing_dir = Path(clothing_crops_dir)
    raw = pd.read_csv(csv_path)
    log.info(f"Raw CSV: {len(raw):,} rows")

    df = raw[raw["skip_reason"].isna()].copy()
    log.info(f"After skip_reason filter: {len(df):,}")

    df = df[df["designer"].isin(TOP_15_DESIGNERS)].reset_index(drop=True)
    log.info(f"After top-15 designer filter: {len(df):,}")

    df["_cloth_exists"] = df["image_id"].apply(
        lambda x: (clothing_dir / f"{x}_clothing.jpg").exists()
    )
    log.info(f"Clothing crops found: {df['_cloth_exists'].sum():,}")
    df = df[df["_cloth_exists"]].drop(columns=["_cloth_exists"]).reset_index(drop=True)
    log.info(f"After clothing-crop filter: {len(df):,}")

    designers_present = sorted(df["designer"].unique().tolist())
    designer_to_idx = {d: i for i, d in enumerate(designers_present)}
    df["designer_label"] = df["designer"].map(designer_to_idx)

    log.info(f"Designers ({len(designer_to_idx)}): {designers_present}")
    log.info(f"Designer distribution:\n{df['designer'].value_counts().to_string()}")

    df.to_csv(cache_path, index=False)
    log.info(f"Saved preprocessed dataset to {cache_path}")
    return df, designer_to_idx


def stratified_split_80_10_10(df, strat_col="designer", seed=42):
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby(strat_col):
        idx = group.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(0.80 * n)
        n_val = int(0.10 * n)
        train_parts.append(group.loc[idx[:n_train]])
        val_parts.append(group.loc[idx[n_train:n_train + n_val]])
        test_parts.append(group.loc[idx[n_train + n_val:]])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# ON-THE-FLY ABSTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def make_silhouette_from_crop(pil_crop):
    """Generate a filled-binary silhouette from a clothing crop.

    Pipeline: grayscale -> threshold at 127 -> fill connected components
    by running morphological closing, then select the largest component
    and fill its interior -> black shape on white background.
    """
    gray = np.array(pil_crop.convert("L"), dtype=np.uint8)
    # Threshold: anything below 127 is "clothing-ish"; the upstream clothing
    # crops from copalette_clothing_crops live on a white background with
    # pixel >= 240 = bg, but we use the looser 127 threshold per spec.
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Fill holes inside the mask via flood fill of the background
    h, w = binary.shape
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary, flood_inv)

    # Morphological closing to clean up fragments
    kernel = np.ones((5, 5), dtype=np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    # Black shape (255 in filled) on white background (0 in filled)
    # -> invert so clothing=0, bg=255 (matches copalette_silhouette_prep)
    silhouette = 255 - filled
    return Image.fromarray(silhouette, mode="L")


def make_edge_from_crop(pil_crop):
    """Generate a Canny edge map from a clothing crop.

    Pipeline: grayscale -> Gaussian blur (3x3, sigma=1.0) -> Canny(50, 150)
    -> invert (black lines on white) -> dilate 2x2. Matches the Canny
    settings used in copalette_edge_prep / copalette_celebrity_scraper.
    """
    gray = np.array(pil_crop.convert("L"), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
    edges = cv2.Canny(blurred, 50, 150)

    # Fallback: if Canny finds essentially nothing, use inverted grayscale
    if int((edges > 0).sum()) < 50:
        edges = 255 - gray

    kernel = np.ones((2, 2), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Invert: black lines on white background
    inverted = 255 - edges
    return Image.fromarray(inverted, mode="L")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class AbstractionDataset(Dataset):
    """Single-stream dataset that yields (img_tensor, label).

    The `condition` argument picks which abstraction to apply:
      - "fullcolor" : clothing crop as-is
      - "grayscale" : clothing crop desaturated, replicated to 3 channels
      - "silhouette": preprocessed silhouette from --silhouette_dir if it
                       exists, otherwise generated on the fly
      - "edge"      : preprocessed edge map from --edge_dir if it exists,
                       otherwise generated on the fly via Canny
    """

    def __init__(self, dataframe, clothing_crops_dir, silhouette_dir, edge_dir,
                 transform, condition):
        assert condition in CONDITIONS, f"Unknown condition {condition}"
        self.df = dataframe.reset_index(drop=True)
        self.clothing_dir = Path(clothing_crops_dir)
        self.silhouette_dir = Path(silhouette_dir) if silhouette_dir else None
        self.edge_dir = Path(edge_dir) if edge_dir else None
        self.transform = transform
        self.condition = condition

    def __len__(self):
        return len(self.df)

    def _load_base(self, image_id):
        path = self.clothing_dir / f"{image_id}_clothing.jpg"
        return Image.open(path).convert("RGB")

    def _load_silhouette(self, image_id, base):
        """Try preprocessed silhouette; fall back to on-the-fly generation."""
        if self.silhouette_dir is not None:
            for suffix in ("_silhouette.jpg", "_silhouette.png"):
                cand = self.silhouette_dir / f"{image_id}{suffix}"
                if cand.exists():
                    try:
                        return Image.open(cand).convert("L")
                    except Exception:
                        pass
        return make_silhouette_from_crop(base)

    def _load_edge(self, image_id, base):
        """Try preprocessed edge map; fall back to on-the-fly generation."""
        if self.edge_dir is not None:
            for suffix in ("_edge.jpg", "_edge.png"):
                cand = self.edge_dir / f"{image_id}{suffix}"
                if cand.exists():
                    try:
                        return Image.open(cand).convert("L")
                    except Exception:
                        pass
        return make_edge_from_crop(base)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        base = self._load_base(image_id)

        if self.condition == "fullcolor":
            img = base
        elif self.condition == "grayscale":
            gray = ImageOps.grayscale(base)
            img = Image.merge("RGB", (gray, gray, gray))
        elif self.condition == "silhouette":
            sil = self._load_silhouette(image_id, base)
            img = Image.merge("RGB", (sil, sil, sil))
        elif self.condition == "edge":
            edge = self._load_edge(image_id, base)
            img = Image.merge("RGB", (edge, edge, edge))
        else:
            raise ValueError(f"Unknown condition {self.condition}")

        img = self.transform(img)
        label = torch.tensor(int(row["designer_label"]), dtype=torch.long)
        return img, label


def make_loader(df, clothing_dir, silhouette_dir, edge_dir, transform,
                condition, batch_size, shuffle):
    ds = AbstractionDataset(df, clothing_dir, silhouette_dir, edge_dir,
                            transform, condition)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
