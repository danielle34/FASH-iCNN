"""
Training loop for designer classification.

AdamW differential LR (backbone 1e-4, head 1e-3, wd=1e-3),
CrossEntropyLoss(label_smoothing=0.1), mixed precision,
patience=15, max_epochs=100, checkpoint resume.
Self-contained — no imports from other copalette modules.
"""

import os
import time
import logging

import torch

log = logging.getLogger("clothing_year")


def make_optimizer(model, backbone_lr=1e-4, head_lr=1e-3, weight_decay=1e-3):
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay)


def train_model(model, train_loader, val_loader, optimizer, device,
                max_epochs=100, patience=15, checkpoint_path=None, no_resume=False):
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    if not no_resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", [])
            log.info(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        except Exception as e:
            log.warning(f"  Failed to resume from checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float("inf")
            patience_counter = 0
            history = []

    model.to(device)

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()

        model.train()
        train_sum, train_n = 0.0, 0
        for batch in train_loader:
            imgs, labels = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = imgs.size(0)
            train_sum += loss.item() * bs
            train_n += bs
        train_loss = train_sum / train_n

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = [b.to(device, non_blocking=True) for b in batch]
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = loss_fn(logits, labels)
                bs = imgs.size(0)
                val_sum += loss.item() * bs
                val_n += bs
        val_loss = val_sum / val_n
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "time_s": elapsed})
        log.info(f"  Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_path:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "history": history,
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    return best_val_loss, history
