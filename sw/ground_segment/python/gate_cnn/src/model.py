"""
model.py
--------
DirectionCNN — lightweight gate detector for Bebop ARMv7.

Input:  [B, 1, 120, 160]  single-channel grayscale (H=120, W=160)
Output: (heading, confidence)
  heading    = tanh(x)    in [-1, 1]   (-1=gate far left, 0=centre, +1=far right)
  confidence = sigmoid(x) in [0,  1]   (probability gate is visible)

Architecture
------------
  Conv1  (1->8,  3x3, pad=1) + ReLU    -> [8,  120, 160]
  Pool1  MaxPool(2x1)                   -> [8,   60, 160]
  Conv2  (8->16, 3x3, pad=1) + ReLU    -> [16,  60, 160]
  Pool2  MaxPool(2x2)                   -> [16,  30,  80]
  Conv3  (16->32,3x3, pad=1) + ReLU    -> [32,  30,  80]
  Pool3  MaxPool(2x2)                   -> [32,  15,  40]
  Conv4  (32->32,3x3, pad=1) + ReLU    -> [32,  15,  40]

  HorizPool  AvgPool(kernel=(15,1))     -> [32,   1,  40]
  Flatten                               -> [1280]

  FC1    Linear(1280->64) + ReLU        -> [64]
  FC2    Linear(64->2)                  -> [2]

  out[0] -> heading    = tanh(out[0])
  out[1] -> confidence = sigmoid(out[1])

WHY horizontal pooling?
-----------------------
Global average pooling collapses ALL spatial dims to a single vector,
destroying left/right position. AvgPool(kernel=(15,1)) pools over HEIGHT
only, keeping all 40 width positions. The FC layer receives 32 channels x
40 positions = 1280 inputs encoding what features are at each horizontal
position — giving the network the spatial info needed to predict heading.
"""

import torch
import torch.nn as nn


class DirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 1->8,  [1, 120, 160] -> [8, 120, 160]
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Pool1: MaxPool(2x1) -> [8, 60, 160]
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Conv2: 8->16,  [8, 60, 160] -> [16, 60, 160]
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Pool2: MaxPool(2x2) -> [16, 30, 80]
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Conv3: 16->32,  [16, 30, 80] -> [32, 30, 80]
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Pool3: MaxPool(2x2) -> [32, 15, 40]
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Conv4: 32->32,  [32, 15, 40] -> [32, 15, 40]  (no pool)
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Pool over HEIGHT only, keep WIDTH (40 positions)
        # [B, 32, 15, 40] -> [B, 32, 1, 40]
        self.horiz_pool = nn.AvgPool2d(kernel_size=(15, 1))

        # After flatten: 32 * 40 = 1280
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 40, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)       # [B, 32, 15, 40]
        x = self.horiz_pool(x)     # [B, 32,  1, 40]
        x = self.head(x)           # [B, 2]
        heading    = torch.tanh(x[:, 0])
        confidence = torch.sigmoid(x[:, 1])
        return heading, confidence


if __name__ == "__main__":
    model = DirectionCNN()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    dummy = torch.randn(1, 1, 120, 160)
    h, c = model(dummy)
    print(f"heading={h.item():+.4f}  confidence={c.item():.4f}")