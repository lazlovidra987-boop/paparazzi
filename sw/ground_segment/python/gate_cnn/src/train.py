"""
train.py
--------
Training script for DirectionCNN.

Usage
-----
    python train.py --root_dir /path/to/data --label_file labels.json

Outputs
-------
  checkpoints/best_model.pth   <- best model by validation loss (use this for export)
  checkpoints/last_model.pth   <- model at final epoch

Loss design
-----------
Two losses are combined:

  1. Confidence loss  (always active)
       BCELoss(predicted_confidence, has_gate)

  2. Heading loss     (only active when has_gate == 1)
       MSELoss(predicted_heading, gt_heading)
       Masked — frames without a gate are excluded so the network
       does not learn nonsense heading values for empty images.

  total_loss = heading_weight * heading_loss + conf_loss

Label vector layout (from dataset.py)
--------------------------------------
  labels[:, 0] = has_gate   (0 or 1)
  labels[:, 1] = heading    (float in [-1, 1])

Class imbalance handling
------------------------
With few gate-visible images the network would learn to always predict
"no gate". We fix this with a WeightedRandomSampler that oversamples
gate-visible images so the effective training ratio is 50/50 each epoch.
"""

import os
import json
import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from model   import DirectionCNN
from dataset import split_dataset, GateDataset, _SubsetDataset


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DirectionCNN")

    # Data
    p.add_argument("--root_dir",    required=True)
    p.add_argument("--label_file",  default="labels.json")
    p.add_argument("--val_split",   type=float, default=0.15)

    # Training
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--heading_weight", type=float, default=1.0,
                   help="Weight on heading MSE loss (default 1.0). "
                        "Increase to 2.0-5.0 if heading is still inaccurate after training.")

    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--seed",           type=int, default=42)

    return p.parse_args()


# ------------------------------------------------------------------
# Weighted sampler — fixes class imbalance
# ------------------------------------------------------------------

def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that oversamples gate-visible images
    so each training epoch sees roughly 50% gate / 50% no-gate batches.
    """
    labels = []
    if isinstance(dataset, _SubsetDataset):
        raw_data = dataset.parent.data
        indices  = dataset.indices
        for i in indices:
            labels.append(int(raw_data[i].get("has_gate", 0)))
    elif isinstance(dataset, GateDataset):
        for entry in dataset.data:
            labels.append(int(entry.get("has_gate", 0)))
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    n_gate    = sum(labels)
    n_no_gate = len(labels) - n_gate
    n_total   = len(labels)

    print(f"  Class balance: {n_gate} gate  |  {n_no_gate} no-gate  "
          f"({100*n_gate/n_total:.1f}% / {100*n_no_gate/n_total:.1f}%)")

    if n_gate == 0 or n_no_gate == 0:
        print("  WARNING: only one class present — sampler disabled.")
        return None

    weight_gate    = n_total / n_gate
    weight_no_gate = n_total / n_no_gate

    sample_weights = torch.tensor(
        [weight_gate if lbl == 1 else weight_no_gate for lbl in labels],
        dtype=torch.float32,
    )

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(dataset),
        replacement = True,
    )
    return sampler


# ------------------------------------------------------------------
# Loss function
# ------------------------------------------------------------------

def compute_loss(
    pred_heading:    torch.Tensor,
    pred_confidence: torch.Tensor,
    labels:          torch.Tensor,
    heading_weight:  float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, heading_loss, conf_loss).

    Label layout from dataset.py:
      labels[:, 0] = has_gate  (0 or 1)
      labels[:, 1] = heading   (float in [-1, 1])
    """
    gt_has_gate = labels[:, 0]   # index 0 = has_gate
    gt_heading  = labels[:, 1]   # index 1 = heading

    # Confidence loss — every sample
    conf_loss = nn.BCELoss()(pred_confidence, gt_has_gate)

    # Heading loss — gate-visible samples only
    gate_mask = gt_has_gate > 0.5
    if gate_mask.sum() > 0:
        heading_loss = nn.MSELoss()(
            pred_heading[gate_mask],
            gt_heading[gate_mask],
        )
    else:
        heading_loss = torch.tensor(0.0, device=pred_heading.device)

    total_loss = heading_weight * heading_loss + conf_loss
    return total_loss, heading_loss, conf_loss


# ------------------------------------------------------------------
# Train / validate one epoch
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, heading_weight):
    model.train()
    t_sum = h_sum = c_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        pred_h, pred_c = model(images)
        loss, h_loss, c_loss = compute_loss(pred_h, pred_c, labels, heading_weight)
        loss.backward()
        optimizer.step()
        t_sum += loss.item()
        h_sum += h_loss.item()
        c_sum += c_loss.item()

    n = len(loader)
    return t_sum / n, h_sum / n, c_sum / n


def validate(model, loader, device, heading_weight):
    model.eval()
    t_sum = h_sum = c_sum = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred_h, pred_c = model(images)
            loss, h_loss, c_loss = compute_loss(pred_h, pred_c, labels, heading_weight)
            t_sum += loss.item()
            h_sum += h_loss.item()
            c_sum += c_loss.item()

            # Accuracy = correct gate/no-gate classification
            gt_has_gate = labels[:, 0]
            correct += ((pred_c > 0.5).float() == gt_has_gate).sum().item()
            total   += labels.size(0)

    n = len(loader)
    return t_sum / n, h_sum / n, c_sum / n, correct / total if total > 0 else 0.0


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset -------------------------------------------------------
    print(f"Loading dataset from: {args.root_dir}")
    train_ds, val_ds = split_dataset(
        root_dir   = args.root_dir,
        label_file = args.label_file,
        val_split  = args.val_split,
        seed       = args.seed,
    )
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # --- Weighted sampler (fixes class imbalance) ----------------------
    print("Building weighted sampler...")
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        sampler     = sampler,
        num_workers = 2,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    # --- Model ---------------------------------------------------------
    model        = DirectionCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    # --- Training loop -------------------------------------------------
    header = (f"{'Epoch':>5}  {'T-Loss':>8}  {'T-Head':>8}  {'T-Conf':>8}  "
              f"{'V-Loss':>8}  {'V-Head':>8}  {'V-Conf':>8}  {'V-Acc':>7}  "
              f"{'LR':>9}  {'Time':>5}")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        t_loss, t_hloss, t_closs = train_one_epoch(
            model, train_loader, optimizer, device, args.heading_weight)

        v_loss, v_hloss, v_closs, v_acc = validate(
            model, val_loader, device, args.heading_weight)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>5}  "
            f"{t_loss:>8.4f}  {t_hloss:>8.4f}  {t_closs:>8.4f}  "
            f"{v_loss:>8.4f}  {v_hloss:>8.4f}  {v_closs:>8.4f}  "
            f"{v_acc:>7.3f}  {current_lr:>9.2e}  {elapsed:>4.1f}s"
        )

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"  -> Best model saved (val_loss={v_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "last_model.pth"))

    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {args.checkpoint_dir}/best_model.pth")
    print(f"\nNext step: python export/export_to_c.py --checkpoint {args.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()