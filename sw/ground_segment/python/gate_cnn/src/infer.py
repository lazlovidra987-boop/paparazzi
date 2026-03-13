"""
infer.py
--------
Run DirectionCNN on a folder of unseen images and visualise predictions.

Usage
-----
    python src/infer.py --checkpoint src/checkpoints/best_model.pth
                        --image_dir  /path/to/unseen/images/

Optional flags
--------------
    --conf_thresh  0.5    confidence threshold to count as "gate detected"
    --save_dir     None   if set, saves annotated images here instead of showing

Display
-------
  GREEN cross  = gate detected, cross placed at predicted gate centre
  RED cross    = low confidence, cross at predicted position (unreliable)
  Yellow line  = image centre (heading = 0, straight ahead)
  Confidence bar on left edge
  SPACE = next image,  q = quit
"""

import os
import sys
import glob
import json
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import DirectionCNN


CNN_INPUT_H = 120
CNN_INPUT_W = 160


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str) -> DirectionCNN:
    model = DirectionCNN().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def draw_result(image_bgr, heading: float, confidence: float,
                conf_thresh: float, gt_heading=None):
    """
    Draw prediction cross and overlay on image.
    Cross is placed at the predicted gate centre:
      cy = heading * (H/2) + H/2   (vertical axis = heading axis)
      cx = W/2                      (horizontal unknown, use centre)
    """
    vis  = image_bgr.copy()
    H, W = vis.shape[:2]

    gate_detected = confidence >= conf_thresh
    colour = (0, 220, 0) if gate_detected else (0, 0, 220)  # green / red

    # Horizontal centre line — heading = 0 reference
    cv2.line(vis, (0, H // 2), (W, H // 2), (60, 60, 60), 1)

    # ── Predicted gate centre cross ──────────────────────────────────────────
    # heading is on the vertical axis (camera rotated 90 deg)
    pred_cy = int(heading * (H / 2.0) + H / 2.0)
    pred_cy = max(0, min(H - 1, pred_cy))
    pred_cx = W // 2   # horizontal position unknown from heading alone

    cross_size  = max(12, H // 10)
    cross_thick = 2

    # Vertical bar of cross
    cv2.line(vis,
             (pred_cx, pred_cy - cross_size),
             (pred_cx, pred_cy + cross_size),
             colour, cross_thick)
    # Horizontal bar of cross
    cv2.line(vis,
             (pred_cx - cross_size, pred_cy),
             (pred_cx + cross_size, pred_cy),
             colour, cross_thick)
    # Circle at centre
    cv2.circle(vis, (pred_cx, pred_cy), 5, colour, -1)

    # ── Ground truth cross (if available) ────────────────────────────────────
    if gt_heading is not None:
        gt_cy = int(gt_heading * (H / 2.0) + H / 2.0)
        gt_cy = max(0, min(H - 1, gt_cy))
        gt_cx = W // 2
        gt_colour = (0, 200, 255)   # yellow = ground truth

        cv2.line(vis,
                 (gt_cx, gt_cy - cross_size),
                 (gt_cx, gt_cy + cross_size),
                 gt_colour, 1)
        cv2.line(vis,
                 (gt_cx - cross_size, gt_cy),
                 (gt_cx + cross_size, gt_cy),
                 gt_colour, 1)
        cv2.circle(vis, (gt_cx, gt_cy), 4, gt_colour, 1)

        # Error line from GT to prediction
        cv2.line(vis, (gt_cx, gt_cy), (pred_cx, pred_cy), (128, 128, 128), 1)

    # ── Confidence bar (left edge) ────────────────────────────────────────────
    bar_h = int(confidence * H)
    cv2.rectangle(vis, (0, H - bar_h), (10, H), colour, -1)

    # ── Text overlays ─────────────────────────────────────────────────────────
    status = "GATE" if gate_detected else "NO GATE"
    cv2.putText(vis, f"{status}  conf={confidence:.2f}",
                (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
    cv2.putText(vis, f"heading={heading:+.3f}",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    if gt_heading is not None:
        err = abs(heading - gt_heading)
        cv2.putText(vis, f"GT={gt_heading:+.3f}  err={err:.3f}",
                    (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    cv2.putText(vis, "SPACE=next  q=quit",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (160, 160, 160), 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    cv2.putText(vis, "GREEN=pred  YELLOW=GT",
                (10, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (160, 160, 160), 1)

    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    device = "cpu"
    print(f"Loading model from: {args.checkpoint}")
    model  = load_model(args.checkpoint, device)
    total  = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    transform = T.Compose([
        T.Resize((CNN_INPUT_H, CNN_INPUT_W)),
        T.Grayscale(),
        T.ToTensor(),
    ])

    # Collect images
    patterns    = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, pat)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in: {args.image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images.")

    # Load ground-truth labels if available
    gt_map     = {}
    label_path = os.path.join(args.image_dir, "labels.json")
    if os.path.exists(label_path):
        with open(label_path) as f:
            labels = json.load(f)
        for entry in labels:
            gt_map[os.path.basename(entry["image"])] = entry
        print(f"Found labels.json — GT crosses shown in yellow.")

    # Output dir
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Saving annotated images to: {args.save_dir}")
    else:
        cv2.namedWindow("Gate CNN Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gate CNN Inference", 400, 700)

    results        = []
    heading_errors = []

    for i, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  [WARN] Could not read {img_path}")
            continue

        # Run model
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        tensor  = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            heading_t, conf_t = model(tensor)

        heading    = heading_t.item()
        confidence = conf_t.item()
        gate_det   = confidence >= args.conf_thresh

        results.append({
            "image":      os.path.basename(img_path),
            "heading":    heading,
            "confidence": confidence,
            "gate":       gate_det,
        })

        # Ground truth
        gt_entry  = gt_map.get(os.path.basename(img_path))
        gt_heading = None
        gt_str     = ""
        if gt_entry and gt_entry.get("has_gate") == 1:
            gt_heading = gt_entry["heading"]
            err        = abs(heading - gt_heading)
            heading_errors.append(err)
            gt_str = f"  GT={gt_heading:+.3f}  err={err:.3f}"

        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}"
              f"  conf={confidence:.3f}  heading={heading:+.3f}"
              f"  {'GATE' if gate_det else 'no gate'}{gt_str}")

        # Draw on full-resolution image
        vis = draw_result(image_bgr, heading, confidence,
                          args.conf_thresh, gt_heading)

        if args.save_dir:
            out_path = os.path.join(args.save_dir,
                                    "pred_" + os.path.basename(img_path))
            cv2.imwrite(out_path, vis)
        else:
            cv2.imshow("Gate CNN Inference", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Quit.")
                break

    if not args.save_dir:
        cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total  = len(results)
    n_gate   = sum(1 for r in results if r["gate"])
    avg_conf = sum(r["confidence"] for r in results) / max(1, n_total)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"  Images processed : {n_total}")
    print(f"  Gate detected    : {n_gate}  ({100*n_gate/max(1,n_total):.1f}%)")
    print(f"  Avg confidence   : {avg_conf:.3f}")
    if heading_errors:
        avg_err = sum(heading_errors) / len(heading_errors)
        print(f"  Avg heading err  : {avg_err:.3f}  "
              f"(on {len(heading_errors)} labelled gate samples)")
    print("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Run DirectionCNN on unseen images with cross overlay")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to best_model.pth")
    p.add_argument("--image_dir",   required=True,
                   help="Folder of images to run inference on")
    p.add_argument("--conf_thresh", type=float, default=0.5,
                   help="Confidence threshold (default 0.5)")
    p.add_argument("--save_dir",    default=None,
                   help="Save annotated images here instead of displaying")
    return p.parse_args()


if __name__ == "__main__":
    main()