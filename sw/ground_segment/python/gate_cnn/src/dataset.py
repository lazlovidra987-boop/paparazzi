# dataset.py
# ------------------------------------------------------------
# PyTorch Dataset for "Gate CNN"
#
# What this dataset returns per sample:
#   - image:  FloatTensor of shape [3, H, W]
#   - label:  FloatTensor of shape [9]
#             [best_gate, u1, v1, u2, v2, u3, v3, u4, v4]
#
# Assumptions:
#   - Labels are already normalized (u,v in [0,1]).
#   - Augmentation has already been applied offline (so we do NOT augment here).
#   - JSON label file is a list of dicts with keys:
#       "image": string path relative to root_dir
#       "has_gate": 0/1
#       "corners": [[u,v],[u,v],[u,v],[u,v]]  (only if has_gate=1, optional if 0)
#
# Example JSON entry:
# {
#   "image": "data/images/train/frame_000123.jpg",
#   "has_gate": 1,
#   "corners": [[0.21,0.31],[0.60,0.30],[0.63,0.78],[0.18,0.80]]
# }
# ------------------------------------------------------------

import os
import json
from typing import Tuple, List, Dict, Any

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GateDataset(Dataset):
    """
    Minimal dataset:
    - loads image
    - resizes to fixed input size for the CNN
    - outputs tensor + label vector
    """

    def __init__(self, root_dir: str, label_file: str, image_size: Tuple[int, int] = (128, 128)):
        """
        Args:
            root_dir: base folder to resolve image paths
            label_file: path to JSON label file (absolute or relative) 
            image_size: (W,H) that the network expects
        """
        self.root_dir = root_dir

        # If label_file is relative, interpret it relative to root_dir.
        if not os.path.isabs(label_file):
            label_file = os.path.join(root_dir, label_file)

        # Load all label entries into memory once (fast + simple).
        with open(label_file, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)

        # Image preprocessing:
        # - resize all images to fixed network input size
        # - convert to tensor in range [0,1]
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])

    def __len__(self) -> int:
        # Number of samples in dataset
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns one sample:
            image_tensor: [3,H,W]
            label_tensor: [9]
        """
        entry = self.data[idx]

        # --------------------------------------------------------
        # 1) Load image
        # --------------------------------------------------------
        # image path from JSON is assumed to be relative to root_dir
        img_path = os.path.join(self.root_dir, entry["image"])

        # Read image as RGB (even if original is grayscale)
        image = Image.open(img_path).convert("RGB")

        # Apply resize + toTensor
        image = self.transform(image)

        # --------------------------------------------------------
        # 2) Build label vector
        # --------------------------------------------------------
        # has_gate is our "best_gate" target during training (0 or 1)
        # Later you can interpret the model output as confidence in [0,1].
        has_gate = int(entry.get("has_gate", 0))

        # If there is a gate: corners must exist (4 points)
        # If no gate: set all corners to 0.
        if has_gate == 1:
            corners = entry["corners"]  # list: [[u,v],[u,v],[u,v],[u,v]]
        else:
            corners = [[0.0, 0.0]] * 4

        # Flatten the corners into a single vector:
        # [u1,v1,u2,v2,u3,v3,u4,v4]
        corners_flat = [
            corners[0][0], corners[0][1],
            corners[1][0], corners[1][1],
            corners[2][0], corners[2][1],
            corners[3][0], corners[3][1],
        ]

        # Final label:
        # [best_gate, u1,v1,u2,v2,u3,v3,u4,v4]
        label = [float(has_gate)] + [float(x) for x in corners_flat]

        label = torch.tensor(label, dtype=torch.float32)

        return image, label