"""
diagnose.py
-----------
Quick check of what the trained model actually predicts.
Run this after training to verify the model learned something useful.

Usage
-----
    python diagnose.py --checkpoint checkpoints/best_model.pth \
                       --root_dir /path/to/data \
                       --label_file labels.json
"""

import os
import json
import argparse
import random

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from model import DirectionCNN
from dataset import CNN_INPUT_W, CNN_INPUT_H


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--root_dir",    required=True)
    p.add_argument("--label_file",  default="labels.json")
    p.add_argument("--n_samples",   type=int, default=20,
                   help="How many images to sample (10 gate + 10 no-gate)")
    return p.parse_args()


def load_image(root_dir, rel_path):
    img = Image.open(os.path.join(root_dir, rel_path)).convert("L")
    img = img.resize((CNN_INPUT_W, CNN_INPUT_H), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0)   # [1, 1, 120, 160]


def main():
    args = get_args()

    # Load model
    model = DirectionCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded: {args.checkpoint}\n")

    # Load labels
    label_path = args.label_file if os.path.isabs(args.label_file) \
                 else os.path.join(args.root_dir, args.label_file)
    with open(label_path) as f:
        data = json.load(f)

    # Split by class
    gate_samples    = [d for d in data if d.get("has_gate") == 1]
    no_gate_samples = [d for d in data if d.get("has_gate") == 0]

    n = args.n_samples // 2
    gate_samples    = random.sample(gate_samples,    min(n, len(gate_samples)))
    no_gate_samples = random.sample(no_gate_samples, min(n, len(no_gate_samples)))

    print(f"{'':=<65}")
    print(f"  {'Image':<30} {'GT':>6} {'GT-Head':>8} {'Pred-C':>8} {'Pred-H':>8} {'OK?':>5}")
    print(f"{'':=<65}")

    correct = 0
    total   = 0

    for entry in gate_samples + no_gate_samples:
        img_tensor = load_image(args.root_dir, entry["image"])
        gt_gate    = entry["has_gate"]
        gt_heading = entry.get("heading", 0.0)

        with torch.no_grad():
            pred_h, pred_c = model(img_tensor)

        pred_conf    = pred_c.item()
        pred_heading = pred_h.item()
        pred_gate    = 1 if pred_conf > 0.5 else 0
        ok           = "✓" if pred_gate == gt_gate else "✗"
        correct     += (pred_gate == gt_gate)
        total       += 1

        name = os.path.basename(entry["image"])[:28]
        print(f"  {name:<30} {gt_gate:>6} {gt_heading:>8.3f} "
              f"{pred_conf:>8.3f} {pred_heading:>8.3f} {ok:>5}")

    print(f"{'':=<65}")
    print(f"  Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print()

    # Summary statistics
    gate_preds    = []
    no_gate_preds = []

    for entry in gate_samples:
        img_tensor = load_image(args.root_dir, entry["image"])
        with torch.no_grad():
            _, pred_c = model(img_tensor)
        gate_preds.append(pred_c.item())

    for entry in no_gate_samples:
        img_tensor = load_image(args.root_dir, entry["image"])
        with torch.no_grad():
            _, pred_c = model(img_tensor)
        no_gate_preds.append(pred_c.item())

    if gate_preds:
        print(f"Gate-visible images    — avg confidence: {sum(gate_preds)/len(gate_preds):.3f}  "
              f"(want > 0.5)")
    if no_gate_preds:
        print(f"No-gate images         — avg confidence: {sum(no_gate_preds)/len(no_gate_preds):.3f}  "
              f"(want < 0.5)")

    print()
    if gate_preds and no_gate_preds:
        avg_gate    = sum(gate_preds)    / len(gate_preds)
        avg_no_gate = sum(no_gate_preds) / len(no_gate_preds)
        separation  = avg_gate - avg_no_gate
        print(f"Confidence separation: {separation:.3f}  ", end="")
        if separation > 0.3:
            print("GOOD — model clearly distinguishes gate vs no-gate")
        elif separation > 0.1:
            print("OK — some separation, more training data would help")
        else:
            print("POOR — model is not distinguishing classes well")
            print("  -> Try collecting more gate-visible images")


if __name__ == "__main__":
    main()