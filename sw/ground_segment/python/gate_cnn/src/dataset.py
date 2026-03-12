"""
dataset.py
----------
PyTorch Dataset for DirectionCNN gate detection.

What each sample contains
--------------------------
  image : FloatTensor  [1, 120, 160]   grayscale, pixels in [0, 1]
  label : FloatTensor  [2]
            label[1] = heading    in [-1, 1]  (-1=far left, 0=centre, +1=far right)
            label[0] = has_gate   0.0 or 1.0

JSON label file format
-----------------------
[
  { "image": "frame_000001.jpg", "has_gate": 1, "heading":  0.35 },
  { "image": "frame_000002.jpg", "has_gate": 0, "heading":  0.0  },
  ...
]

Heading is computed from the gate centre pixel as:
    heading = (gate_center_x_pixels - W/2) / (W/2)

Augmentations (training only)
------------------------------
The following are applied randomly each time a sample is loaded.
Augmentations that move the gate spatially also update the heading label.

  Transform               Heading update needed?
  ----------------------  ----------------------
  Horizontal flip         YES  — negate heading
  Rotation ±15°           YES  — rotate gate centre point
  Zoom / crop             NO   — crop is gate-centred, keeps heading
  Brightness/contrast     NO   — pixel only
  Gaussian noise          NO   — pixel only
  Motion blur             NO   — pixel only
  Perspective warp        YES  — warp gate centre point through same matrix
"""

import os
import json
import random
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


# ------------------------------------------------------------------
# Network input resolution — must match global_pool in model.py
# ------------------------------------------------------------------
CNN_INPUT_W = 160
CNN_INPUT_H = 120


# ------------------------------------------------------------------
# Augmentation probabilities — tweak here if needed
# ------------------------------------------------------------------
P_FLIP        = 0.5
P_ROTATE      = 0.5
P_ZOOM        = 0.4
P_NOISE       = 0.4
P_BLUR        = 0.3
P_PERSPECTIVE = 0.3

ROTATE_MAX_DEG   = 15.0   # ± degrees
ZOOM_MAX_FACTOR  = 0.20   # crop up to 20% of each side
NOISE_STD        = 0.04   # gaussian noise std (image in [0,1])
BLUR_MAX_KERNEL  = 5      # motion blur kernel max size (pixels, odd numbers only)
PERSP_DISTORTION = 0.10   # how much the corners can shift (fraction of image size)


class GateDataset(Dataset):
    """
    Loads gate images and returns (image_tensor, label_tensor).

    Args:
        root_dir   : base folder; image paths in JSON are relative to this.
        label_file : JSON label file (absolute, or relative to root_dir).
        augment    : if True apply all augmentations (training only).
    """

    def __init__(self, root_dir: str, label_file: str, augment: bool = False):
        self.root_dir = root_dir
        self.augment  = augment

        if not os.path.isabs(label_file):
            label_file = os.path.join(root_dir, label_file)

        with open(label_file, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)

        # Base transform: always applied (train + val)
        self.base_transform = T.Compose([
            T.Resize((CNN_INPUT_H, CNN_INPUT_W),
                     interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),   # [0,255] PIL -> [0,1] FloatTensor [1,H,W]
        ])

        # Colour jitter (brightness + contrast only — image is grayscale)
        self.jitter = T.ColorJitter(brightness=0.35, contrast=0.35)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.data[idx]

        # 1. Load grayscale image
        img_path = os.path.join(self.root_dir, entry["image"])
        image    = Image.open(img_path).convert("L")

        # 2. Resize to CNN input size first so all heading maths is in
        #    a consistent [CNN_INPUT_W x CNN_INPUT_H] coordinate space.
        image = image.resize((CNN_INPUT_W, CNN_INPUT_H), Image.BILINEAR)

        # 3. Read labels
        has_gate = float(entry.get("has_gate", 0))
        heading  = float(entry.get("heading",  0.0))
        heading  = max(-1.0, min(1.0, heading))

        # Convert heading -> gate centre pixel (for transforms that need it)
        # cx is in [0, CNN_INPUT_W-1]
        cx = _heading_to_cx(heading, CNN_INPUT_W)
        cy = CNN_INPUT_H / 2.0   # we only know horizontal position from heading

        # 4. Augmentation
        if self.augment:
            image, cx, cy = self._augment(image, cx, cy, has_gate)
            # Recompute heading from (possibly moved) gate centre
            if has_gate:
                heading = _cx_to_heading(cx, CNN_INPUT_W)
                heading = max(-1.0, min(1.0, heading))

        # 5. To tensor (does NOT resize again — already the right size)
        image = TF.to_tensor(image)   # [1, 120, 160], values in [0,1]

        label = torch.tensor([has_gate, heading], dtype=torch.float32)
        return image, label

    # ------------------------------------------------------------------
    # Augmentation — returns (image, new_cx, new_cy)
    # ------------------------------------------------------------------

    def _augment(
        self,
        image: Image.Image,
        cx: float,
        cy: float,
        has_gate: float,
    ) -> Tuple[Image.Image, float, float]:
        W, H = CNN_INPUT_W, CNN_INPUT_H

        # --- Horizontal flip -------------------------------------------
        # New heading = -heading  (exact enough for W=160)
        if random.random() < P_FLIP:
            image = TF.hflip(image)
            if has_gate:
                cx = W - 1 - cx

        # --- Rotation ±15° --------------------------------------------
        # Gate centre rotates around image centre.
        # cx_new = cos(a)*(cx - W/2) - sin(a)*(cy - H/2) + W/2
        if random.random() < P_ROTATE:
            angle = random.uniform(-ROTATE_MAX_DEG, ROTATE_MAX_DEG)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR,
                              fill=0)
            if has_gate:
                a     = math.radians(angle)
                dx    = cx - W / 2.0
                dy    = cy - H / 2.0
                cx    = math.cos(a) * dx - math.sin(a) * dy + W / 2.0
                cy    = math.sin(a) * dx + math.cos(a) * dy + H / 2.0

        # --- Zoom / crop (gate-centred) --------------------------------
        # We crop a random margin from each side symmetrically so the gate
        # centre stays at the same relative position -> heading unchanged.
        # The crop is then resized back to CNN_INPUT size.
        if random.random() < P_ZOOM:
            margin = random.uniform(0.0, ZOOM_MAX_FACTOR)
            left   = int(W * margin)
            top    = int(H * margin)
            right  = W - left
            bottom = H - top
            # Ensure at least 10px remain
            if right - left > 10 and bottom - top > 10:
                image = TF.crop(image, top, left, bottom - top, right - left)
                image = image.resize((W, H), Image.BILINEAR)
                # Remap gate centre into cropped coordinate space
                if has_gate:
                    cx = (cx - left) / (right - left) * W
                    cy = (cy - top)  / (bottom - top) * H

        # --- Brightness / contrast jitter ------------------------------
        # Pixel-only — no heading change needed.
        image = self.jitter(image)

        # --- Gaussian noise --------------------------------------------
        # Pixel-only — no heading change needed.
        if random.random() < P_NOISE:
            image = _add_gaussian_noise(image, std=NOISE_STD)

        # --- Motion blur -----------------------------------------------
        # Pixel-only — no heading change needed.
        if random.random() < P_BLUR:
            image = _apply_motion_blur(image, max_kernel=BLUR_MAX_KERNEL)

        # --- Perspective warp ------------------------------------------
        # We build the warp matrix manually so we can map the gate centre
        # through the same transform.
        if random.random() < P_PERSPECTIVE:
            image, cx, cy = _apply_perspective(image, cx, cy, has_gate,
                                               distortion=PERSP_DISTORTION)

        # Clamp gate centre to image bounds
        cx = max(0.0, min(float(W - 1), cx))
        cy = max(0.0, min(float(H - 1), cy))

        return image, cx, cy


# ------------------------------------------------------------------
# Pixel-level augmentation helpers
# ------------------------------------------------------------------

def _add_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    """Add zero-mean Gaussian noise to a grayscale PIL image."""
    arr   = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
    arr   = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def _apply_motion_blur(image: Image.Image, max_kernel: int) -> Image.Image:
    """Apply horizontal or vertical motion blur to simulate fast movement."""
    # Pick a random odd kernel size between 3 and max_kernel
    k = random.choice([k for k in range(3, max_kernel + 1, 2)])

    # Randomly choose horizontal or vertical blur
    if random.random() < 0.5:
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k   # horizontal
    else:
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[:, k // 2] = 1.0 / k   # vertical

    from PIL import ImageFilter
    # PIL's built-in kernel filter
    arr    = np.array(image, dtype=np.float32)
    from scipy.ndimage import convolve
    blurred = convolve(arr, kernel, mode='reflect')
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred, mode="L")


def _apply_perspective(
    image:      Image.Image,
    cx:         float,
    cy:         float,
    has_gate:   float,
    distortion: float,
) -> Tuple[Image.Image, float, float]:
    """
    Apply a small random perspective warp using OpenCV.
    The gate centre point is mapped through the same homography.
    """
    try:
        import cv2
    except ImportError:
        # OpenCV not available — skip perspective warp silently
        return image, cx, cy

    W, H = CNN_INPUT_W, CNN_INPUT_H
    d    = distortion

    # Source corners (the four corners of the image)
    src = np.float32([
        [0,     0    ],
        [W - 1, 0    ],
        [W - 1, H - 1],
        [0,     H - 1],
    ])

    # Destination corners — perturb each corner randomly by up to d*size
    dst = src.copy()
    dst[0] += np.random.uniform(0, d * W, 2)
    dst[1] += np.random.uniform(-d * W, 0, (1,)).tolist() + \
               np.random.uniform(0, d * H, (1,)).tolist()
    dst[2] += np.random.uniform(-d * W, 0, (1,)).tolist() + \
               np.random.uniform(-d * H, 0, (1,)).tolist()
    dst[3] += np.random.uniform(0, d * W, (1,)).tolist() + \
               np.random.uniform(-d * H, 0, (1,)).tolist()

    M = cv2.getPerspectiveTransform(src, dst)

    # Warp image
    arr     = np.array(image)
    warped  = cv2.warpPerspective(arr, M, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    image   = Image.fromarray(warped, mode="L")

    # Warp gate centre through the same homography
    if has_gate:
        pt      = np.float32([[[cx, cy]]])
        pt_warped = cv2.perspectiveTransform(pt, M)
        cx      = float(pt_warped[0, 0, 0])
        cy      = float(pt_warped[0, 0, 1])

    return image, cx, cy


# ------------------------------------------------------------------
# Heading <-> pixel helpers
# ------------------------------------------------------------------

def _heading_to_cx(heading: float, W: int) -> float:
    """Convert normalised heading [-1,1] to gate centre x pixel."""
    return heading * (W / 2.0) + W / 2.0


def _cx_to_heading(cx: float, W: int) -> float:
    """Convert gate centre x pixel to normalised heading [-1,1]."""
    return (cx - W / 2.0) / (W / 2.0)


# ------------------------------------------------------------------
# Train / val split utility
# ------------------------------------------------------------------

class _SubsetDataset(Dataset):
    def __init__(self, parent: GateDataset, indices: List[int]):
        self.parent  = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.parent[self.indices[idx]]


def split_dataset(
    root_dir:   str,
    label_file: str,
    val_split:  float = 0.15,
    seed:       int   = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into augmented train subset and clean val subset.

    Returns:
        train_dataset, val_dataset
    """
    full_train = GateDataset(root_dir, label_file, augment=True)
    full_val   = GateDataset(root_dir, label_file, augment=False)

    n       = len(full_train)
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val

    rng     = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    return (
        _SubsetDataset(full_train, indices[:n_train]),
        _SubsetDataset(full_val,   indices[n_train:]),
    )


# ------------------------------------------------------------------
# Sanity check — run this file directly to verify augmentations
# ------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, sys

    print("Testing dataset with all augmentations...")

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir)

        labels = []
        for i in range(20):
            img = Image.new("L", (640, 480), color=100 + i * 5)
            fname = f"frame_{i:04d}.jpg"
            img.save(os.path.join(img_dir, fname))
            labels.append({
                "image":    f"images/{fname}",
                "has_gate": i % 2,
                "heading":  round((i - 10) * 0.08, 3),
            })

        lf = os.path.join(tmpdir, "labels.json")
        with open(lf, "w") as f:
            json.dump(labels, f)

        train_ds, val_ds = split_dataset(tmpdir, "labels.json", val_split=0.2)

        errors = []
        for i in range(len(train_ds)):
            img, lbl = train_ds[i]
            if img.shape != (1, CNN_INPUT_H, CNN_INPUT_W):
                errors.append(f"sample {i}: wrong shape {img.shape}")
            if not (-1.0 <= lbl[0].item() <= 1.0):
                errors.append(f"sample {i}: heading out of range {lbl[0].item():.3f}")
            if lbl[1].item() not in (0.0, 1.0):
                errors.append(f"sample {i}: has_gate not 0/1: {lbl[1].item()}")

        if errors:
            print("ERRORS:")
            for e in errors:
                print(f"  {e}")
            sys.exit(1)

        img, lbl = train_ds[0]
        print(f"Train samples  : {len(train_ds)}")
        print(f"Val   samples  : {len(val_ds)}")
        print(f"Image shape    : {tuple(img.shape)}   (expected [1, {CNN_INPUT_H}, {CNN_INPUT_W}])")
        print(f"Label shape    : {tuple(lbl.shape)}   (expected [2])")
        print(f"Heading range  : OK (all in [-1, 1])")
        print(f"has_gate range : OK (all 0.0 or 1.0)")
        print("dataset.py OK")