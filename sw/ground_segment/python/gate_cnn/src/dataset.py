"""
dataset.py
----------
PyTorch Dataset for DirectionCNN gate detection.

What each sample contains
--------------------------
  image : FloatTensor  [1, 120, 160]   grayscale, pixels in [0, 1]
  label : FloatTensor  [2]
            label[0] = gate_commitment   0.0 or 1.0
            label[1] = heading    in [-1, 1]

CAMERA ORIENTATION NOTE
-----------------------
The Bebop front camera is physically rotated 90 degrees.
Images are portrait (H=120, W=160 after resize).
The drone's LEFT/RIGHT corresponds to the VERTICAL axis of the image.

  heading = (gate_cy - H/2) / (H/2)

  heading = -1.0  ->  gate at TOP    of image  ->  drone's LEFT
  heading =  0.0  ->  gate at CENTRE of image  ->  straight ahead
  heading = +1.0  ->  gate at BOTTOM of image  ->  drone's RIGHT

All augmentations that move the gate spatially update cy (not cx),
and heading is recomputed from cy after augmentation.

JSON label file format
-----------------------
[
  { "image": "frame_000001.jpg", "gate_commitment": 1, "heading":  0.35 },
  { "image": "frame_000002.jpg", "gate_commitment": 0, "heading":  0.0  },
  ...
]

Augmentations (training only)
------------------------------
  Transform               Heading update?   Axis
  ----------------------  ---------------   ----
  Vertical flip           YES               cy = H-1-cy  -> negate heading
  Horizontal flip         NO                cx only, heading unaffected
  Rotation +/-15deg       YES               rotate (cx,cy) around centre
  Zoom / crop             YES               remap cy into cropped coords
  Brightness/contrast     NO                pixel only
  Gaussian noise          NO                pixel only
  Motion blur             NO                pixel only
  Perspective warp        YES               warp (cx,cy) through homography
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
from PIL import Image


# ------------------------------------------------------------------
# Network input resolution — must match horiz_pool in model.py
# ------------------------------------------------------------------
CNN_INPUT_W = 160
CNN_INPUT_H = 120


# ------------------------------------------------------------------
# Augmentation probabilities
# ------------------------------------------------------------------
P_VFLIP       = 0.5    # vertical flip  — negates heading (camera rotated)
P_HFLIP       = 0.5    # horizontal flip — does NOT affect heading
P_ROTATE      = 0.5    # rotation +/-15 deg
P_ZOOM        = 0.4    # random crop + resize
P_NOISE       = 0.4    # gaussian noise
P_BLUR        = 0.3    # motion blur
P_PERSPECTIVE = 0.3    # perspective warp

ROTATE_MAX_DEG   = 15.0
ZOOM_MAX_FACTOR  = 0.20
NOISE_STD        = 0.04
BLUR_MAX_KERNEL  = 5
PERSP_DISTORTION = 0.10


class GateDataset(Dataset):
    """
    Loads gate images and returns (image_tensor, label_tensor).

    Args:
        root_dir   : base folder; image paths in JSON are relative to this.
        label_file : JSON label file (absolute, or relative to root_dir).
        augment    : if True, apply augmentations (training only).
    """

    def __init__(self, root_dir: str, label_file: str, augment: bool = False):
        self.root_dir = root_dir
        self.augment  = augment

        if not os.path.isabs(label_file):
            label_file = os.path.join(root_dir, label_file)

        with open(label_file, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)

        # Colour jitter applied during augmentation
        self.jitter = T.ColorJitter(brightness=0.35, contrast=0.35)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.data[idx]

        # 1. Load grayscale image and resize to CNN input size
        img_path = os.path.join(self.root_dir, entry["image"])
        image    = Image.open(img_path).convert("L")
        image    = image.resize((CNN_INPUT_W, CNN_INPUT_H), Image.BILINEAR)

        # 2. Read labels
        gate_commitment = float(entry.get("gate_commitment", 0))
        heading  = float(entry.get("heading",  0.0))
        heading  = max(-1.0, min(1.0, heading))

        # 3. Convert heading -> gate centre pixel on VERTICAL axis
        #    cy is in [0, CNN_INPUT_H-1]
        #    cx is unknown from labels — use image centre
        cy = _heading_to_cy(heading, CNN_INPUT_H)
        cx = CNN_INPUT_W / 2.0

        # 4. Augmentation (training only)
        if self.augment:
            image, cx, cy = self._augment(image, cx, cy, gate_commitment)
            # Recompute heading from moved gate centre (vertical axis)
            if gate_commitment:
                heading = _cy_to_heading(cy, CNN_INPUT_H)
                heading = max(-1.0, min(1.0, heading))

        # 5. To tensor [1, H, W] in [0, 1]
        image = TF.to_tensor(image)

        label = torch.tensor([gate_commitment, heading], dtype=torch.float32)
        return image, label

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(
        self,
        image:    Image.Image,
        cx:       float,
        cy:       float,
        gate_commitment: float,
    ) -> Tuple[Image.Image, float, float]:

        W, H = CNN_INPUT_W, CNN_INPUT_H

        # --- Vertical flip --------------------------------------------
        # Camera is rotated: up/down in image = left/right for drone.
        # Flipping vertically negates heading.
        if random.random() < P_VFLIP:
            image = TF.vflip(image)
            if gate_commitment:
                cy = (H - 1) - cy   # cy reflects around H/2 -> heading negated

        # --- Horizontal flip ------------------------------------------
        # Left/right in image has NO effect on heading (camera rotated).
        # Still useful as augmentation to prevent overfitting to background.
        if random.random() < P_HFLIP:
            image = TF.hflip(image)
            # cx flips but heading is unaffected
            if gate_commitment:
                cx = (W - 1) - cx

        # --- Rotation +/-15 deg --------------------------------------
        # Both cx and cy rotate around image centre.
        if random.random() < P_ROTATE:
            angle = random.uniform(-ROTATE_MAX_DEG, ROTATE_MAX_DEG)
            image = TF.rotate(image, angle,
                              interpolation=TF.InterpolationMode.BILINEAR,
                              fill=0)
            if gate_commitment:
                a  = math.radians(angle)
                dx = cx - W / 2.0
                dy = cy - H / 2.0
                cx = math.cos(a) * dx - math.sin(a) * dy + W / 2.0
                cy = math.sin(a) * dx + math.cos(a) * dy + H / 2.0

        # --- Zoom / crop ---------------------------------------------
        # Symmetric crop so gate centre moves proportionally.
        if random.random() < P_ZOOM:
            margin = random.uniform(0.0, ZOOM_MAX_FACTOR)
            left   = int(W * margin)
            top    = int(H * margin)
            right  = W - left
            bottom = H - top
            if right - left > 10 and bottom - top > 10:
                image = TF.crop(image, top, left, bottom - top, right - left)
                image = image.resize((W, H), Image.BILINEAR)
                if gate_commitment:
                    cx = (cx - left)   / (right  - left)  * W
                    cy = (cy - top)    / (bottom - top)   * H

        # --- Brightness / contrast -----------------------------------
        image = self.jitter(image)

        # --- Gaussian noise ------------------------------------------
        if random.random() < P_NOISE:
            image = _add_gaussian_noise(image, std=NOISE_STD)

        # --- Motion blur ---------------------------------------------
        if random.random() < P_BLUR:
            image = _apply_motion_blur(image, max_kernel=BLUR_MAX_KERNEL)

        # --- Perspective warp ----------------------------------------
        if random.random() < P_PERSPECTIVE:
            image, cx, cy = _apply_perspective(
                image, cx, cy, gate_commitment, distortion=PERSP_DISTORTION)

        # Clamp to image bounds
        cx = max(0.0, min(float(W - 1), cx))
        cy = max(0.0, min(float(H - 1), cy))

        return image, cx, cy


# ------------------------------------------------------------------
# Pixel-level augmentation helpers
# ------------------------------------------------------------------

def _add_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    arr   = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
    arr   = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def _apply_motion_blur(image: Image.Image, max_kernel: int) -> Image.Image:
    k = random.choice([k for k in range(3, max_kernel + 1, 2)])
    if random.random() < 0.5:
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k   # horizontal
    else:
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[:, k // 2] = 1.0 / k   # vertical
    arr = np.array(image, dtype=np.float32)
    from scipy.ndimage import convolve
    blurred = convolve(arr, kernel, mode='reflect')
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    return Image.fromarray(blurred, mode="L")


def _apply_perspective(
    image:      Image.Image,
    cx:         float,
    cy:         float,
    gate_commitment:   float,
    distortion: float,
) -> Tuple[Image.Image, float, float]:
    try:
        import cv2
    except ImportError:
        return image, cx, cy

    W, H = CNN_INPUT_W, CNN_INPUT_H
    d    = distortion

    src = np.float32([
        [0,     0    ],
        [W - 1, 0    ],
        [W - 1, H - 1],
        [0,     H - 1],
    ])
    dst = src.copy()
    dst[0] += np.random.uniform(0, d * W, 2)
    dst[1] += np.array([np.random.uniform(-d * W, 0),
                         np.random.uniform(0, d * H)])
    dst[2] += np.array([np.random.uniform(-d * W, 0),
                         np.random.uniform(-d * H, 0)])
    dst[3] += np.array([np.random.uniform(0, d * W),
                         np.random.uniform(-d * H, 0)])

    M      = cv2.getPerspectiveTransform(src, dst)
    arr    = np.array(image)
    warped = cv2.warpPerspective(arr, M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    image  = Image.fromarray(warped, mode="L")

    if gate_commitment:
        pt        = np.float32([[[cx, cy]]])
        pt_warped = cv2.perspectiveTransform(pt, M)
        cx        = float(pt_warped[0, 0, 0])
        cy        = float(pt_warped[0, 0, 1])

    return image, cx, cy


# ------------------------------------------------------------------
# Heading <-> pixel helpers  (VERTICAL axis — camera rotated 90 deg)
# ------------------------------------------------------------------

def _heading_to_cy(heading: float, H: int) -> float:
    """Normalised heading [-1,1] -> gate centre y pixel [0, H-1]."""
    return heading * (H / 2.0) + H / 2.0


def _cy_to_heading(cy: float, H: int) -> float:
    """Gate centre y pixel -> normalised heading [-1,1]."""
    return (cy - H / 2.0) / (H / 2.0)


# ------------------------------------------------------------------
# Train / val split
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
# Sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, sys

    print("Testing dataset with all augmentations...")

    with tempfile.TemporaryDirectory() as tmpdir:
        labels = []
        for i in range(20):
            img = Image.new("L", (640, 480), color=100 + i * 5)
            fname = f"frame_{i:04d}.jpg"
            img.save(os.path.join(tmpdir, fname))
            labels.append({
                "image":    fname,
                "gate_commitment": i % 2,
                "heading":  round((i - 10) * 0.08, 3),
            })

        lf = os.path.join(tmpdir, "labels.json")
        with open(lf, "w") as f:
            json.dump(labels, f)

        train_ds, val_ds = split_dataset(tmpdir, "labels.json", val_split=0.2)

        errors = []
        for i in range(len(train_ds)):
            img_t, lbl = train_ds[i]
            if img_t.shape != (1, CNN_INPUT_H, CNN_INPUT_W):
                errors.append(f"sample {i}: wrong shape {img_t.shape}")
            if not (-1.0 <= lbl[1].item() <= 1.0):
                errors.append(f"sample {i}: heading out of range {lbl[1].item():.3f}")
            if lbl[0].item() not in (0.0, 1.0):
                errors.append(f"sample {i}: gate_commitment not 0/1: {lbl[0].item()}")

        if errors:
            print("ERRORS:")
            for e in errors: print(f"  {e}")
            sys.exit(1)

        img_t, lbl = train_ds[0]
        print(f"Train samples : {len(train_ds)}")
        print(f"Val   samples : {len(val_ds)}")
        print(f"Image shape   : {tuple(img_t.shape)}  (want [1, {CNN_INPUT_H}, {CNN_INPUT_W}])")
        print(f"Label shape   : {tuple(lbl.shape)}    (want [2])")
        print(f"label[0]=gate_commitment, label[1]=heading — both in range")
        print("dataset.py OK")