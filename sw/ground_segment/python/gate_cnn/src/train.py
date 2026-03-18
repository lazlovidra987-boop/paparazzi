"""
train.py
--------
Training script for DirectionGateNet.

Usage
-----
    python train.py --root_dir /path/to/images --label_file /path/to/labels_mike.json

Outputs
-------
  checkpoints/best_model_dronet.pth   <- best model by validation loss
  checkpoints/last_model_dronet.pth   <- model at final epoch
"""

import os
import re
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from model_dronet import DirectionGateNet
from dataset import GateDataset, _SubsetDataset


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DirectionGateNet")

    # Data
    p.add_argument("--root_dir", required=True, help="Folder containing the images")
    p.add_argument("--label_file", default="labels_mike.json", help="Path to JSON label file")
    p.add_argument("--val_split", type=float, default=0.15)

    # Split mode
    p.add_argument(
        "--split_mode",
        choices=["chronological", "random"],
        default="chronological",
        help="Chronological split is better for video-like data",
    )

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--heading_weight", type=float, default=2.0)
    p.add_argument("--gate_weight", type=float, default=1.0)

    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default="runs/directiongatenet_dronet")

    return p.parse_args()


def resolve_label_file(label_file: str) -> str:
    """
    Resolve label file robustly from:
    - absolute path
    - current working directory
    - src/
    - gate_cnn root
    """
    candidates = []

    if os.path.isabs(label_file):
        candidates.append(label_file)
    else:
        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gate_cnn_root = os.path.dirname(script_dir)

        candidates.append(os.path.join(cwd, label_file))
        candidates.append(os.path.join(script_dir, label_file))
        candidates.append(os.path.join(gate_cnn_root, label_file))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Could not find label file '{label_file}'. Tried:\n" + "\n".join(candidates)
    )


def resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    """
    Resolve checkpoint directory relative to gate_cnn root,
    not relative to wherever the terminal currently is.
    """
    if os.path.isabs(checkpoint_dir):
        return checkpoint_dir

    script_dir = os.path.dirname(os.path.abspath(__file__))
    gate_cnn_root = os.path.dirname(script_dir)
    return os.path.abspath(os.path.join(gate_cnn_root, checkpoint_dir))


def frame_number_from_name(name: str) -> int:
    base = os.path.basename(name)
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else -1


def split_dataset_custom(
    root_dir: str,
    label_file: str,
    val_split: float,
    seed: int,
    split_mode: str = "chronological",
):
    full_ds = GateDataset(root_dir=root_dir, label_file=label_file)

    n = len(full_ds)
    if n < 2:
        raise ValueError(f"Dataset too small: {n} samples")

    if split_mode == "chronological":
        indices = list(range(n))
        indices.sort(key=lambda i: frame_number_from_name(full_ds.data[i]["image"]))
    else:
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n, generator=g).tolist()

    n_val = max(1, int(round(val_split * n)))
    n_train = n - n_val

    if n_train < 1:
        raise ValueError(
            f"Split invalid: n={n}, val_split={val_split} gives train={n_train}, val={n_val}"
        )

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = _SubsetDataset(full_ds, train_indices)
    val_ds = _SubsetDataset(full_ds, val_indices)

    return train_ds, val_ds


def make_weighted_sampler(dataset) -> WeightedRandomSampler | None:
    labels = []

    if isinstance(dataset, _SubsetDataset):
        raw_data = dataset.parent.data
        indices = dataset.indices
        for i in indices:
            labels.append(int(raw_data[i].get("gate_commitment", 0)))

    elif isinstance(dataset, GateDataset):
        for entry in dataset.data:
            labels.append(int(entry.get("gate_commitment", 0)))

    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    n_gate = sum(labels)
    n_no_gate = len(labels) - n_gate
    n_total = len(labels)

    print(
        f"  Class balance: {n_gate} gate  |  {n_no_gate} no-gate  "
        f"({100*n_gate/max(1,n_total):.1f}% / {100*n_no_gate/max(1,n_total):.1f}%)"
    )

    if n_gate == 0 or n_no_gate == 0:
        print("  WARNING: only one class present — sampler disabled.")
        return None

    weight_gate = n_total / n_gate
    weight_no_gate = n_total / n_no_gate

    sample_weights = torch.tensor(
        [weight_gate if lbl == 1 else weight_no_gate for lbl in labels],
        dtype=torch.float32,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


def compute_loss(
    pred_heading: torch.Tensor,
    pred_gate: torch.Tensor,
    labels: torch.Tensor,
    heading_weight: float = 1.0,
    gate_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_gate_commitment = labels[:, 0]
    gt_heading = labels[:, 1]

    gate_loss = nn.BCELoss()(pred_gate, gt_gate_commitment)

    gate_mask = gt_gate_commitment > 0.5
    if gate_mask.sum() > 0:
        heading_loss = nn.MSELoss()(
            pred_heading[gate_mask],
            gt_heading[gate_mask],
        )
    else:
        heading_loss = torch.tensor(0.0, device=pred_heading.device)

    total_loss = heading_weight * heading_loss + gate_weight * gate_loss
    return total_loss, heading_loss, gate_loss


def train_one_epoch(model, loader, optimizer, device, heading_weight, gate_weight):
    model.train()
    t_sum = h_sum = g_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred_h, pred_g = model(images)

        loss, h_loss, g_loss = compute_loss(
            pred_h, pred_g, labels, heading_weight, gate_weight
        )

        loss.backward()
        optimizer.step()

        t_sum += loss.item()
        h_sum += h_loss.item()
        g_sum += g_loss.item()

    n = len(loader)
    return t_sum / n, h_sum / n, g_sum / n


def validate(model, loader, device, heading_weight, gate_weight):
    model.eval()
    t_sum = h_sum = g_sum = 0.0
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            pred_h, pred_g = model(images)
            loss, h_loss, g_loss = compute_loss(
                pred_h, pred_g, labels, heading_weight, gate_weight
            )

            t_sum += loss.item()
            h_sum += h_loss.item()
            g_sum += g_loss.item()

            gt_gate_commitment = labels[:, 0]
            correct += ((pred_g > 0.5).float() == gt_gate_commitment).sum().item()
            total += labels.size(0)

    n = len(loader)
    return t_sum / n, h_sum / n, g_sum / n, correct / total if total > 0 else 0.0


def log_model_weights(writer: SummaryWriter, model: torch.nn.Module, epoch: int):
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)
        if param.grad is not None:
            writer.add_histogram(f"grads/{name}", param.grad.detach().cpu(), epoch)


def log_first_layer_kernels(writer: SummaryWriter, model: torch.nn.Module, epoch: int):
    try:
        first_conv = model.stem[0]
        if isinstance(first_conv, nn.Conv2d):
            weights = first_conv.weight.detach().cpu()
            writer.add_images("kernels/stem_conv", weights, epoch)
    except Exception as e:
        print(f"  Warning: could not log first-layer kernels: {e}")


def log_sample_images(writer: SummaryWriter, loader: DataLoader, epoch: int):
    try:
        images, labels = next(iter(loader))
        images = images[:8].detach().cpu()
        writer.add_images("samples/train_images", images, epoch)
    except Exception as e:
        print(f"  Warning: could not log sample images: {e}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_file = resolve_label_file(args.label_file)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)

    print(f"Using label file: {label_file}")
    print(f"Checkpoint dir:   {checkpoint_dir}")

    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logs: {args.log_dir}")

    print(f"Loading images from: {args.root_dir}")
    train_ds, val_ds = split_dataset_custom(
        root_dir=args.root_dir,
        label_file=label_file,
        val_split=args.val_split,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    print(f"Split mode: {args.split_mode}")

    print("Building weighted sampler...")
    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = DirectionGateNet(output_mode="probability").to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print("Model: DirectionGateNet from model_dronet.py")

    writer.add_text("run_info/model", "DirectionGateNet(output_mode='probability')")
    writer.add_text("run_info/device", str(device))
    writer.add_text("run_info/parameters", f"{total_params:,}")
    writer.add_text("run_info/heading_weight", str(args.heading_weight))
    writer.add_text("run_info/gate_weight", str(args.gate_weight))
    writer.add_text("run_info/lr", str(args.lr))
    writer.add_text("run_info/split_mode", str(args.split_mode))
    writer.add_text("run_info/label_file", str(label_file))
    writer.add_text("run_info/checkpoint_dir", str(checkpoint_dir))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    log_sample_images(writer, train_loader, epoch=0)

    header = (
        f"{'Epoch':>5}  {'T-Loss':>8}  {'T-Head':>8}  {'T-Gate':>8}  "
        f"{'V-Loss':>8}  {'V-Head':>8}  {'V-Gate':>8}  {'V-Acc':>7}  "
        f"{'LR':>9}  {'Time':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        t_loss, t_hloss, t_gloss = train_one_epoch(
            model, train_loader, optimizer, device,
            args.heading_weight, args.gate_weight
        )

        v_loss, v_hloss, v_gloss, v_acc = validate(
            model, val_loader, device,
            args.heading_weight, args.gate_weight
        )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>5}  "
            f"{t_loss:>8.4f}  {t_hloss:>8.4f}  {t_gloss:>8.4f}  "
            f"{v_loss:>8.4f}  {v_hloss:>8.4f}  {v_gloss:>8.4f}  "
            f"{v_acc:>7.3f}  {current_lr:>9.2e}  {elapsed:>4.1f}s"
        )

        writer.add_scalar("loss/train_total", t_loss, epoch)
        writer.add_scalar("loss/train_heading", t_hloss, epoch)
        writer.add_scalar("loss/train_gate", t_gloss, epoch)

        writer.add_scalar("loss/val_total", v_loss, epoch)
        writer.add_scalar("loss/val_heading", v_hloss, epoch)
        writer.add_scalar("loss/val_gate", v_gloss, epoch)

        writer.add_scalar("metrics/val_gate_accuracy", v_acc, epoch)
        writer.add_scalar("train/learning_rate", current_lr, epoch)
        writer.add_scalar("train/epoch_time_sec", elapsed, epoch)

        log_model_weights(writer, model, epoch)
        log_first_layer_kernels(writer, model, epoch)

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_path = os.path.join(checkpoint_dir, "best_model_dronet.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved to: {best_path}")

    last_path = os.path.join(checkpoint_dir, "last_model_dronet.pth")
    torch.save(model.state_dict(), last_path)
    print(f"Last model saved to: {last_path}")

    writer.close()

    print("\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {os.path.join(checkpoint_dir, 'best_model_dronet.pth')}")
    print("TensorBoard   : tensorboard --logdir=runs")


if __name__ == "__main__":
    main()