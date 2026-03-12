"""
autolabel.py
------------
Automatically labels Bebop front-camera JPEG images for gate detection.

IMPORTANT — Camera orientation
-------------------------------
The Bebop front camera is physically rotated 90 degrees. Images are stored
as portrait (H=520, W=240). The drone's left/right corresponds to the
VERTICAL axis of the image (cy), NOT the horizontal axis (cx).

Heading is therefore computed as:
    heading = (cy - image_height / 2) / (image_height / 2)

  heading = -1.0  ->  gate at TOP    of image  ->  drone's LEFT
  heading =  0.0  ->  gate at CENTRE of image  ->  straight ahead
  heading = +1.0  ->  gate at BOTTOM of image  ->  drone's RIGHT

Detection approach
------------------
A thick square foam gate appears in images as a large bright rectangular
contour with a hollow dark centre. We detect it using adaptive thresholding
and contour filtering (area, aspect ratio, solidity).

Manual verification UI
----------------------
Each image is shown with the detected gate highlighted.

  LEFT CLICK anywhere on the image to manually place/move the gate centre.
  The heading line updates immediately when you click.

  Keyboard:
    y / ENTER   accept current label (auto-detected or manually placed)
    n           mark as no gate (has_gate=0, heading=0.0)
    s           skip this image entirely (not added to JSON)
    q           quit and save all labels collected so far

Visual feedback
---------------
  Yellow horizontal line  = image centre (= "straight ahead")
  Green dot + line        = auto-detected gate centre
  Orange dot + line       = manually clicked gate centre

Output JSON (ready for dataset.py)
-----------------------------------
  [
    { "image": "frame_000001.jpg", "has_gate": 1, "heading":  0.35 },
    { "image": "frame_000002.jpg", "has_gate": 0, "heading":  0.0  },
    ...
  ]

Usage
-----
    python autolabel.py --image_dir /path/to/images --output labels.json
    python autolabel.py --image_dir /path/to/images --max_images 200
    python autolabel.py --image_dir /path/to/images --min_area 800
"""

import os
import sys
import json
import argparse
import glob

import cv2
import numpy as np


# ------------------------------------------------------------------
# Detection defaults (tunable via CLI)
# ------------------------------------------------------------------
DEFAULT_MIN_AREA     = 1500
DEFAULT_MAX_AREA     = 120000
DEFAULT_ASPECT_MIN   = 0.55
DEFAULT_ASPECT_MAX   = 1.45
DEFAULT_SOLIDITY_MIN = 0.20
DEFAULT_SOLIDITY_MAX = 0.85

WINDOW_NAME = "Gate Labeller"


# ------------------------------------------------------------------
# Mouse state
# ------------------------------------------------------------------
class MouseState:
    def __init__(self):
        self.cx      = None
        self.cy      = None
        self.clicked = False


mouse = MouseState()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disp_w, disp_h, img_w, img_h = param
        scale_x = img_w / disp_w if disp_w > 0 else 1.0
        scale_y = img_h / disp_h if disp_h > 0 else 1.0
        mouse.cx      = int(x * scale_x)
        mouse.cy      = int(y * scale_y)
        mouse.clicked = True


# ------------------------------------------------------------------
# Gate detector
# ------------------------------------------------------------------

def detect_gate(image_bgr, args):
    H, W = image_bgr.shape[:2]

    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=4,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dict(found=False, cx=None, cy=None, bbox=None, contour=None)

    best_score = -1.0
    best       = dict(found=False, cx=None, cy=None, bbox=None, contour=None)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < args.min_area or area > args.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if not (args.aspect_min <= aspect <= args.aspect_max):
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if not (args.solidity_min <= solidity <= args.solidity_max):
            continue

        cx_cnt = x + w // 2
        cy_cnt = y + h // 2

        # Favour gates near VERTICAL centre (heading axis for rotated camera)
        score = (area / (H * W)) * 10.0 + aspect + \
                (1.0 - abs(cy_cnt - H / 2) / (H / 2)) * 0.5

        if score > best_score:
            best_score = score
            best = dict(found=True, cx=cx_cnt, cy=cy_cnt,
                        bbox=(x, y, w, h), contour=cnt)

    return best


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def compute_heading(cy, image_height):
    """
    Normalise gate centre y to [-1, 1].
    Camera is rotated 90 deg: left/right in world = up/down in image.
      cy = 0       -> heading -1.0  (gate to drone's LEFT)
      cy = H/2     -> heading  0.0  (gate straight ahead)
      cy = H       -> heading +1.0  (gate to drone's RIGHT)
    """
    return (cy - image_height / 2.0) / (image_height / 2.0)


def draw_overlay(image_bgr, detection, cx_active, cy_active, heading, manual):
    vis  = image_bgr.copy()
    H, W = vis.shape[:2]
    colour = (0, 140, 255) if manual else (0, 255, 0)  # orange=manual, green=auto

    # Horizontal centre line = heading 0 reference
    cv2.line(vis, (0, H // 2), (W, H // 2), (255, 255, 0), 1)

    # Auto-detected contour
    if detection["found"] and detection["contour"] is not None:
        cv2.drawContours(vis, [detection["contour"]], -1, (0, 160, 0), 1)
        x, y, w, h = detection["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 160), 1)

    # Active gate centre
    if cx_active is not None and cy_active is not None:
        cv2.circle(vis, (cx_active, cy_active), 8,  colour, -1)
        cv2.circle(vis, (cx_active, cy_active), 14, colour,  2)

        # Vertical line from horizontal centre to gate cy (shows heading deviation)
        cv2.line(vis, (cx_active, H // 2), (cx_active, cy_active), colour, 2)

        src = "MANUAL" if manual else "AUTO"
        cv2.putText(vis, f"{src}  heading: {heading:+.3f}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(vis, "GATE FOUND",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    else:
        cv2.putText(vis, "NO GATE — click to place, or press  n",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(vis,
                "CLICK=set centre   y/ENTER=accept   n=no gate   s=skip   q=quit",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return vis


def _save(labels, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


# ------------------------------------------------------------------
# Main labelling loop
# ------------------------------------------------------------------

def main():
    args = get_args()

    patterns    = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
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
    print("Camera is rotated 90deg — heading measured on VERTICAL axis (cy).")
    print("  Gate at TOP    of image = heading -1.0 (drone's LEFT)")
    print("  Gate at BOTTOM of image = heading +1.0 (drone's RIGHT)")
    print("CLICK to place gate centre.  y/ENTER=accept  n=no gate  s=skip  q=quit")
    print("-" * 60)

    labels          = []
    labelled_images = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            labels = json.load(f)
        labelled_images = {entry["image"] for entry in labels}
        print(f"Resuming — {len(labels)} labels already saved.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 400, 700)  # portrait window matches image shape

    accepted = skipped = 0

    for i, img_path in enumerate(image_paths):
        rel_path = os.path.relpath(img_path, args.image_dir)
        if rel_path in labelled_images:
            continue

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  [WARN] Could not read {img_path}, skipping.")
            continue

        img_H, img_W = image_bgr.shape[:2]

        detection = detect_gate(image_bgr, args)
        cx_active = detection["cx"]
        cy_active = detection["cy"]
        manual    = False

        mouse.clicked = False
        mouse.cx = mouse.cy = None

        status = (f"AUTO heading={compute_heading(cy_active, img_H):+.3f}"
                  if cy_active is not None else "NOT FOUND")
        print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}  ->  {status}")

        while True:

            if mouse.clicked:
                cx_active = mouse.cx
                cy_active = mouse.cy
                manual    = True
                mouse.clicked = False
                print(f"  -> Click: cy={cy_active}  "
                      f"heading={compute_heading(cy_active, img_H):+.3f}")

            heading = compute_heading(cy_active, img_H) if cy_active is not None else 0.0

            rect   = cv2.getWindowImageRect(WINDOW_NAME)
            disp_w = rect[2] if rect[2] > 0 else 400
            disp_h = rect[3] if rect[3] > 0 else 700
            cv2.setMouseCallback(
                WINDOW_NAME, mouse_callback,
                param=(disp_w, disp_h, img_W, img_H),
            )

            vis = draw_overlay(image_bgr, detection, cx_active, cy_active, heading, manual)
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord('y'), 13):
                has_gate = 1 if cy_active is not None else 0
                labels.append({
                    "image":    rel_path,
                    "has_gate": has_gate,
                    "heading":  round(float(heading), 6),
                })
                accepted += 1
                src = "manual" if manual else "auto"
                print(f"  -> Accepted ({src})  has_gate={has_gate}  heading={heading:+.3f}")
                break

            elif key == ord('n'):
                labels.append({"image": rel_path, "has_gate": 0, "heading": 0.0})
                accepted += 1
                print("  -> NO GATE")
                break

            elif key == ord('s'):
                skipped += 1
                print("  -> Skipped")
                break

            elif key == ord('q'):
                print("\nQuitting...")
                _save(labels, args.output)
                print(f"Saved {len(labels)} labels to {args.output}")
                cv2.destroyAllWindows()
                sys.exit(0)

        if len(labels) % 20 == 0 and len(labels) > 0:
            _save(labels, args.output)

    _save(labels, args.output)
    cv2.destroyAllWindows()

    gate_count = sum(1 for l in labels if l["has_gate"] == 1)
    print("\n" + "=" * 60)
    print(f"Done!  Accepted={accepted}  Skipped={skipped}")
    print(f"Gate visible: {gate_count}/{len(labels)} "
          f"({100 * gate_count / max(1, len(labels)):.1f}%)")
    print(f"Saved to: {args.output}")
    print("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",    required=True)
    p.add_argument("--output",       default="labels.json")
    p.add_argument("--max_images",   type=int,   default=0)
    p.add_argument("--min_area",     type=int,   default=DEFAULT_MIN_AREA)
    p.add_argument("--max_area",     type=int,   default=DEFAULT_MAX_AREA)
    p.add_argument("--aspect_min",   type=float, default=DEFAULT_ASPECT_MIN)
    p.add_argument("--aspect_max",   type=float, default=DEFAULT_ASPECT_MAX)
    p.add_argument("--solidity_min", type=float, default=DEFAULT_SOLIDITY_MIN)
    p.add_argument("--solidity_max", type=float, default=DEFAULT_SOLIDITY_MAX)
    return p.parse_args()


if __name__ == "__main__":
    main()