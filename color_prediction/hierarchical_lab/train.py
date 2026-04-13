"""
Training utilities for copalette_hierarchical_lab.

  - train_family_css_classifier: trains one Stage-2 head over the FROZEN
    multimodal trunk for a single Berlin-Kay family. Operates on cached
    feature tensors so each family trains in seconds.

  - train_lab_regressor: trains the Stage-3 constrained LAB regressor on
    cached (face+clothing) features and TRUE CSS centroids. Loss is MSE
    between the predicted LAB and the true LAB. The trained regressor is
    later evaluated under both oracle (true CSS) and pipeline (predicted
    CSS) routing.
"""

import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import FamilyCSSClassifier, LABRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2: per-family CSS classifier
# ---------------------------------------------------------------------------
def train_family_css_classifier(
    family_name, fusion_dim,
    train_features, train_labels,
    val_features, val_labels,
    num_classes, device, checkpoint_path,
    max_epochs=50, patience=10, batch_size=512,
    no_resume=False,
):
    """
    Train a small CSS classifier over cached features for one BK family.
    Returns (model, best_val_loss).
    """
    model = FamilyCSSClassifier(fusion_dim, num_classes).to(device)

    if not no_resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("num_classes") == num_classes:
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"  [{family_name}] loaded checkpoint, val_loss="
                        f"{ckpt.get('best_val_loss', float('nan')):.4f}")
            return model, float(ckpt.get("best_val_loss", float("nan")))
        else:
            logger.warning(f"  [{family_name}] checkpoint num_classes mismatch, retraining")

    # Single-class degenerate case: nothing to learn, just record
    if num_classes < 2:
        logger.info(f"  [{family_name}] only 1 CSS class — skipping training")
        torch.save({
            "model_state": model.state_dict(),
            "best_val_loss": 0.0,
            "num_classes": num_classes,
        }, checkpoint_path)
        return model, 0.0

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    n_train = train_features.size(0)
    n_val = val_features.size(0)

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x = train_features[idx]
            y = train_labels[idx]
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                x = val_features[i:i + batch_size]
                y = val_labels[i:i + batch_size]
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "best_val_loss": best_val_loss,
                "num_classes": num_classes,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= patience:
            logger.info(f"  [{family_name}] epoch {epoch:3d}: "
                        f"train={train_loss:.4f} val={val_loss:.4f} "
                        f"({elapsed:.1f}s)")

        if patience_counter >= patience:
            logger.info(f"  [{family_name}] early stop at epoch {epoch}")
            break

    # Load best
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    return model, best_val_loss


# ---------------------------------------------------------------------------
# Stage 3: constrained LAB regressor
# ---------------------------------------------------------------------------
def train_lab_regressor(
    train_features, train_centroids, train_lab,
    val_features, val_centroids, val_lab,
    in_dim, device, checkpoint_path,
    max_offset=(10.0, 15.0, 15.0),
    max_epochs=50, patience=10, batch_size=512, lr=1e-3,
    no_resume=False,
):
    """
    Train the constrained LAB regression head on cached features.
    `*_centroids` are the per-sample CSS centroids that the regression
    offset is added to.

    Returns (model, best_val_loss).
    """
    model = LABRegressor(in_dim=in_dim, max_offset=max_offset).to(device)

    if not no_resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_val_loss = float(ckpt.get("best_val_loss", float("nan")))
        logger.info(f"  Loaded LAB regressor checkpoint, val_loss={best_val_loss:.4f}")
        return model, best_val_loss

    train_features = train_features.to(device)
    train_centroids = train_centroids.to(device)
    train_lab = train_lab.to(device)
    val_features = val_features.to(device)
    val_centroids = val_centroids.to(device)
    val_lab = val_lab.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    n_train = train_features.size(0)
    n_val = val_features.size(0)

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        train_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x = train_features[idx]
            c = train_centroids[idx]
            y = train_lab[idx]
            optimizer.zero_grad()
            pred = model(x, c)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                x = val_features[i:i + batch_size]
                c = val_centroids[i:i + batch_size]
                y = val_lab[i:i + batch_size]
                pred = model(x, c)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= patience:
            logger.info(f"  LAB regressor epoch {epoch:3d}: "
                        f"train_mse={train_loss:.4f} val_mse={val_loss:.4f} "
                        f"({elapsed:.1f}s)")

        if patience_counter >= patience:
            logger.info(f"  LAB regressor early stop at epoch {epoch}")
            break

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    return model, best_val_loss
