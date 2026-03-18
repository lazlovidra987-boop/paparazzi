import os
import json
import argparse
import random

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from model_dronet import DirectionGateNet


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--label_file", default="labels_mike.json")
    p.add_argument("--n_samples", type=int, default=20)
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


def load_image(root_dir, rel_path):
    img_path = os.path.join(root_dir, rel_path)
    img = Image.open(img_path).convert("L")
    img = img.resize((160, 120), Image.BILINEAR)
    return TF.to_tensor(img).unsqueeze(0)


def main():
    args = get_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    label_file = resolve_label_file(args.label_file)
    root_dir = os.path.abspath(args.root_dir)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Using labels:      {label_file}")
    print(f"Using root_dir:    {root_dir}")
    print("Model source:      model_dronet.py -> DirectionGateNet\n")

    model = DirectionGateNet(output_mode="probability")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    with open(label_file, "r") as f:
        data = json.load(f)

    gate_samples = [d for d in data if d["gate_commitment"] == 1]
    no_gate_samples = [d for d in data if d["gate_commitment"] == 0]

    n = args.n_samples // 2
    gate_samples = random.sample(gate_samples, min(n, len(gate_samples)))
    no_gate_samples = random.sample(no_gate_samples, min(n, len(no_gate_samples)))

    print("=" * 78)
    print(f"{'Image':<30} {'GT':>6} {'GT-H':>8} {'Pred-C':>8} {'Pred-H':>8} {'OK?':>5}")
    print("=" * 78)

    correct = 0
    total = 0

    for entry in gate_samples + no_gate_samples:
        img_tensor = load_image(root_dir, entry["image"])

        gt_gate = entry["gate_commitment"]
        gt_heading = entry["heading"]

        with torch.no_grad():
            pred_h, pred_c = model(img_tensor)

        pred_conf = pred_c.item()
        pred_heading = pred_h.item()
        pred_gate = 1 if pred_conf > 0.5 else 0

        ok = "✓" if pred_gate == gt_gate else "✗"

        correct += int(pred_gate == gt_gate)
        total += 1

        name = os.path.basename(entry["image"])[:28]
        print(
            f"{name:<30} {gt_gate:>6} {gt_heading:>8.3f} "
            f"{pred_conf:>8.3f} {pred_heading:>8.3f} {ok:>5}"
        )

    print("=" * 78)
    print(f"Accuracy: {correct}/{total} = {100*correct/max(1,total):.1f}%\n")

    gate_preds = []
    no_gate_preds = []

    for entry in gate_samples:
        img_tensor = load_image(root_dir, entry["image"])
        with torch.no_grad():
            _, pred_c = model(img_tensor)
        gate_preds.append(pred_c.item())

    for entry in no_gate_samples:
        img_tensor = load_image(root_dir, entry["image"])
        with torch.no_grad():
            _, pred_c = model(img_tensor)
        no_gate_preds.append(pred_c.item())

    if gate_preds:
        print(f"Gate images avg conf: {sum(gate_preds)/len(gate_preds):.3f}")
    if no_gate_preds:
        print(f"No-gate avg conf:    {sum(no_gate_preds)/len(no_gate_preds):.3f}")

    if gate_preds and no_gate_preds:
        separation = (sum(gate_preds)/len(gate_preds)) - (sum(no_gate_preds)/len(no_gate_preds))
        print(f"Separation: {separation:.3f}")

        if separation > 0.3:
            print("GOOD separation")
        elif separation > 0.1:
            print("OK separation")
        else:
            print("POOR separation")


if __name__ == "__main__":
    main()