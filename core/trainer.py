"""
core/trainer.py
───────────────
Training loop with:
  • Progress callbacks (for Streamlit live updates)
  • Val accuracy tracking
  • Early stopping
  • Checkpoint saving
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        # Handle both tuple (imgs, labels) and dict batches
        if isinstance(batch, dict):
            imgs = batch['image']
            labels = batch['label']
        else:
            imgs, labels = batch

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        # Handle both tuple (imgs, labels) and dict batches
        if isinstance(batch, dict):
            imgs = batch['image']
            labels = batch['label']
        else:
            imgs, labels = batch

        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def train(model, train_loader, val_loader, config: dict,
          device: str, progress_callback=None, status_callback=None):
    """
    Full training loop.

    Args:
        model             : BiasAwareCNN instance
        train_loader      : DataLoader
        val_loader        : DataLoader
        config (dict)     : {epochs, lr, weight_decay, patience, save_path}
        device            : 'cuda' | 'cpu'
        progress_callback : fn(epoch, total_epochs, train_loss, val_loss, val_acc)
        status_callback   : fn(message_str)

    Returns:
        history (dict): {train_loss, val_loss, val_acc} lists
    """
    epochs       = config.get("epochs",       10)
    lr           = config.get("lr",           1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    patience     = config.get("patience",     3)
    save_path    = config.get("save_path",    "best_model.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        msg = (f"Epoch {epoch:02d}/{epochs} | "
               f"TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.3f} | "
               f"ValLoss={val_loss:.4f}  ValAcc={val_acc:.3f} | "
               f"Time={elapsed:.1f}s")

        if status_callback:
            status_callback(msg)

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, val_loss, val_acc)

        # ── Checkpoint ──────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":    val_acc,
                "history":    history,
            }, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if status_callback:
                    status_callback(f"Early stopping at epoch {epoch}.")
                break

    return history, best_val_acc
