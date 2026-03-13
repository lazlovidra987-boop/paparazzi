"""
autolabel.py
------------
Semi-automatic label tool for DirectionGateNet training.

Labels
------
Each image gets:
  - heading
  - gate_commitment

gate_commitment = 1
    The drone is close / aligned enough that it should now commit
    to flying through the gate.

gate_commitment = 0
    The drone should still stay in general navigation logic.

Heading
-------
The Bebop front camera is rotated by 90 degrees, so left/right steering is
measured on the VERTICAL image axis (cy):

    heading = (cy - image_height / 2) / (image_height / 2)

  heading = -1.0  -> target at TOP    of image -> drone should go LEFT
  heading =  0.0  -> target at CENTRE of image -> straight
  heading = +1.0  -> target at BOTTOM of image -> drone should go RIGHT

Workflow
--------
The tool proposes:
  1. a target point for heading
  2. a gate_commitment value

You only correct when needed.

Controls
--------
  ENTER / y   accept current label
  LEFT CLICK  move heading target manually
  c           toggle gate_commitment (0 <-> 1)
  a           restore automatic target
  n           set gate_commitment=0 and heading=0.0
  s           skip image
  q           quit and save progress

Output JSON
-----------
[
  {
    "image": "frame_000001.jpg",
    "gate_commitment": 1,
    "heading": 0.214532
  },
  ...
]

Usage
-----
    python autolabel.py --image_dir /path/to/images --output labels.json
    python autolabel.py --image_dir /path/to/images --max_images 200
"""

import os
import sys
import json
import glob
import argparse

import cv2
import numpy as np


DEFAULT_MIN_AREA = 1500
DEFAULT_MAX_AREA = 120000
DEFAULT_ASPECT_MIN = 0.55
DEFAULT_ASPECT_MAX = 1.45
DEFAULT_SOLIDITY_MIN = 0.20
DEFAULT_SOLIDITY_MAX = 0.85

DEFAULT_COMMIT_AREA_FRAC = 0.10
DEFAULT_COMMIT_CENTER_TOL = 0.35
DEFAULT_COMMIT_FULLY_INSIDE_MARGIN = 4

WINDOW_NAME = "Gate Commitment Labeller"


class MouseState:
    def __init__(self):
        self.cx = None
        self.cy = None
        self.clicked = False


mouse = MouseState()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disp_w, disp_h, img_w, img_h = param
        scale_x = img_w / disp_w if disp_w > 0 else 1.0
        scale_y = img_h / disp_h if disp_h > 0 else 1.0
        mouse.cx = int(x * scale_x)
        mouse.cy = int(y * scale_y)
        mouse.clicked = True


def detect_gate(image_bgr, args):
    H, W = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=4,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "found": False,
            "cx": None,
            "cy": None,
            "bbox": None,
            "contour": None,
            "score": 0.0,
        }

    best_score = -1.0
    best = {
        "found": False,
        "cx": None,
        "cy": None,
        "bbox": None,
        "contour": None,
        "score": 0.0,
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < args.min_area or area > args.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
        if not (args.aspect_min <= aspect <= args.aspect_max):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if not (args.solidity_min <= solidity <= args.solidity_max):
            continue

        cx_cnt = x + w // 2
        cy_cnt = y + h // 2

        area_score = area / float(H * W)
        center_score = 1.0 - abs(cy_cnt - H / 2.0) / max(1.0, H / 2.0)

        score = area_score * 10.0 + aspect + 0.5 * center_score

        if score > best_score:
            best_score = score
            best = {
                "found": True,
                "cx": cx_cnt,
                "cy": cy_cnt,
                "bbox": (x, y, w, h),
                "contour": cnt,
                "score": score,
            }

    return best


def compute_heading(cy, image_height):
    heading = (cy - image_height / 2.0) / (image_height / 2.0)
    return float(np.clip(heading, -1.0, 1.0))


def propose_gate_commitment(detection, image_shape, args):
    H, W = image_shape[:2]

    if not detection["found"] or detection["bbox"] is None:
        return 0

    x, y, w, h = detection["bbox"]
    cy = detection["cy"]

    bbox_area_frac = (w * h) / float(H * W)
    center_offset = abs(cy - H / 2.0) / max(1.0, H / 2.0)

    margin = args.commit_inside_margin
    fully_inside = (
        x >= margin and
        y >= margin and
        (x + w) <= (W - margin) and
        (y + h) <= (H - margin)
    )

    big_enough = bbox_area_frac >= args.commit_area_frac
    centered_enough = center_offset <= args.commit_center_tol

    return int(big_enough and centered_enough and fully_inside)


def draw_overlay(
    image_bgr,
    detection,
    cx_active,
    cy_active,
    heading,
    manual,
    gate_commitment,
):
    vis = image_bgr.copy()
    H, W = vis.shape[:2]

    target_colour = (0, 140, 255) if manual else (0, 255, 0)
    commit_colour = (0, 220, 0) if gate_commitment == 1 else (0, 0, 255)

    cv2.line(vis, (0, H // 2), (W, H // 2), (255, 255, 0), 1)

    if detection["found"] and detection["contour"] is not None:
        cv2.drawContours(vis, [detection["contour"]], -1, (0, 160, 0), 1)
        x, y, w, h = detection["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 160), 1)

    if cx_active is not None and cy_active is not None:
        cv2.circle(vis, (cx_active, cy_active), 8, target_colour, -1)
        cv2.circle(vis, (cx_active, cy_active), 14, target_colour, 2)
        cv2.line(vis, (cx_active, H // 2), (cx_active, cy_active), target_colour, 2)

        src = "MANUAL" if manual else "AUTO"
        cv2.putText(
            vis,
            f"{src} heading: {heading:+.3f}",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            target_colour,
            2,
        )
    else:
        cv2.putText(
            vis,
            "NO TARGET POINT",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        vis,
        f"gate_commitment: {gate_commitment}",
        (10, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        commit_colour,
        2,
    )

    cv2.putText(
        vis,
        "CLICK=target  ENTER/y=accept  c=toggle commit  a=auto  n=none  s=skip  q=quit",
        (10, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )

    return vis


def save_labels(labels, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output", default="labels.json")
    p.add_argument("--max_images", type=int, default=0)

    p.add_argument("--min_area", type=int, default=DEFAULT_MIN_AREA)
    p.add_argument("--max_area", type=int, default=DEFAULT_MAX_AREA)
    p.add_argument("--aspect_min", type=float, default=DEFAULT_ASPECT_MIN)
    p.add_argument("--aspect_max", type=float, default=DEFAULT_ASPECT_MAX)
    p.add_argument("--solidity_min", type=float, default=DEFAULT_SOLIDITY_MIN)
    p.add_argument("--solidity_max", type=float, default=DEFAULT_SOLIDITY_MAX)

    p.add_argument("--commit_area_frac", type=float, default=DEFAULT_COMMIT_AREA_FRAC)
    p.add_argument("--commit_center_tol", type=float, default=DEFAULT_COMMIT_CENTER_TOL)
    p.add_argument("--commit_inside_margin", type=int, default=DEFAULT_COMMIT_FULLY_INSIDE_MARGIN)

    return p.parse_args()


def main():
    args = get_args()

    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, pat)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in: {args.image_dir}")
        sys.exit(1)

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} images.")
    print("Camera is rotated 90deg — heading uses image Y coordinate.")
    print("ENTER/y=accept  click=target  c=toggle commitment  a=auto  n=none  s=skip  q=quit")
    print("-" * 70)

    labels = []
    labelled_images = set()

    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            labels = json.load(f)
        labelled_images = {entry["image"] for entry in labels}
        print(f"Resuming — {len(labels)} labels already saved.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 400, 700)

    accepted = 0
    skipped = 0

    for i, img_path in enumerate(image_paths):
        rel_path = os.path.relpath(img_path, args.image_dir)
        if rel_path in labelled_images:
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue

        img_H, img_W = image_bgr.shape[:2]

        detection = detect_gate(image_bgr, args)

        auto_cx = detection["cx"]
        auto_cy = detection["cy"]

        cx_active = auto_cx
        cy_active = auto_cy
        manual = False

        gate_commitment = propose_gate_commitment(detection, image_bgr.shape, args)

        mouse.clicked = False
        mouse.cx = None
        mouse.cy = None

        if cy_active is not None:
            auto_heading = compute_heading(cy_active, img_H)
            print(
                f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}"
                f"  ->  AUTO heading={auto_heading:+.3f}, gate_commitment={gate_commitment}"
            )
        else:
            print(
                f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}"
                f"  ->  NO TARGET, gate_commitment=0"
            )

        while True:
            if mouse.clicked:
                cx_active = mouse.cx
                cy_active = mouse.cy
                manual = True
                mouse.clicked = False

                print(
                    f"  -> Click: cy={cy_active}, heading={compute_heading(cy_active, img_H):+.3f}"
                )

            heading = compute_heading(cy_active, img_H) if cy_active is not None else 0.0

            rect = cv2.getWindowImageRect(WINDOW_NAME)
            disp_w = rect[2] if rect[2] > 0 else 400
            disp_h = rect[3] if rect[3] > 0 else 700
            cv2.setMouseCallback(
                WINDOW_NAME,
                mouse_callback,
                param=(disp_w, disp_h, img_W, img_H),
            )

            vis = draw_overlay(
                image_bgr=image_bgr,
                detection=detection,
                cx_active=cx_active,
                cy_active=cy_active,
                heading=heading,
                manual=manual,
                gate_commitment=gate_commitment,
            )
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord("y"), 13):
                labels.append({
                    "image": rel_path,
                    "gate_commitment": int(gate_commitment),
                    "heading": round(float(heading), 6),
                })
                accepted += 1
                src = "manual" if manual else "auto"
                print(
                    f"  -> Accepted ({src})  gate_commitment={gate_commitment}  heading={heading:+.3f}"
                )
                break

            elif key == ord("c"):
                gate_commitment = 1 - int(gate_commitment)
                print(f"  -> gate_commitment toggled to {gate_commitment}")

            elif key == ord("a"):
                cx_active = auto_cx
                cy_active = auto_cy
                manual = False
                print("  -> Restored auto target")

            elif key == ord("n"):
                labels.append({
                    "image": rel_path,
                    "gate_commitment": 0,
                    "heading": 0.0,
                })
                accepted += 1
                print("  -> gate_commitment=0, heading=0")
                break

            elif key == ord("s"):
                skipped += 1
                print("  -> Skipped")
                break

            elif key == ord("q"):
                print("\nQuitting...")
                save_labels(labels, args.output)
                print(f"Saved {len(labels)} labels to {args.output}")
                cv2.destroyAllWindows()
                sys.exit(0)

        if len(labels) % 20 == 0 and len(labels) > 0:
            save_labels(labels, args.output)

    save_labels(labels, args.output)
    cv2.destroyAllWindows()

    n_commit = sum(1 for l in labels if int(l.get("gate_commitment", 0)) == 1)

    print("\n" + "=" * 70)
    print(f"Done! Accepted={accepted}  Skipped={skipped}")
    print(f"Commitment=1: {n_commit}/{len(labels)} ({100 * n_commit / max(1, len(labels)):.1f}%)")
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()