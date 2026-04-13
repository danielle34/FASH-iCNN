"""
Training loop for copalette_clothing_decade.
Single-image classifier with mixed precision, early stopping, checkpoint resume.
"""

import os
import time
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


def make_optimizer(model, backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-3):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _resume(model, optimizer, scheduler, ckpt_path, device, no_resume):
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    if not no_resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", [])
        logger.info(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss, patience_counter, history


def train_classifier(model, train_loader, val_loader, device, checkpoint_path,
                     max_epochs=80, patience=10, no_resume=False,
                     label_smoothing=0.1):
    """Train an (img) -> logits classifier."""
    model = model.to(device)
    optimizer = make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler() if device.type == "cuda" else None

    start_epoch, best_val_loss, patience_counter, history = _resume(
        model, optimizer, scheduler, checkpoint_path, device, no_resume)

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        model.train()
        train_loss, n_train = 0.0, 0
        for img, label in train_loader:
            img = img.to(device); label = label.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    logits = model(img)
                    loss = loss_fn(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(img)
                loss = loss_fn(logits, label)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * img.size(0)
            n_train += img.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device); label = label.to(device)
                if scaler:
                    with autocast():
                        logits = model(img)
                        loss = loss_fn(logits, label)
                else:
                    logits = model(img)
                    loss = loss_fn(logits, label)
                val_loss += loss.item() * img.size(0)
                n_val += img.size(0)
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "time_s": elapsed})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "history": history,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter >= patience:
            logger.info(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f} ({elapsed:.1f}s)")

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    return best_val_loss, history
