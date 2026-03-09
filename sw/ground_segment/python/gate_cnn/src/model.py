# model.py
# ------------------------------------------------------------
# CNN for horizontal flight-direction prediction
#
# Goal:
# Given a camera image, predict ONE value in [0,1]
# that tells the drone where it should fly horizontally.
#
# Output:
#   0.0 = far left
#   1.0 = far right
#
# Design choices:
# - Input uses only the Y channel (1 channel total)
# - We do NOT crop away width information inside the model
# - We reduce HEIGHT faster than WIDTH in the first layers
#   because horizontal information is more important
# - We use a small network so it has a realistic chance
#   of running on drone hardware
# ------------------------------------------------------------

import torch
import torch.nn as nn

class DirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # FEATURE EXTRACTOR
        # ----------------------------------------------------
        # This part of the network learns visual features:
        # edges, shapes, openings, gates, free space, etc.
        #
        # Important idea:
        # We do NOT want to lose horizontal information too early,
        # because the final output is a horizontal direction.
        #
        # Therefore:
        # - first pooling layer: reduce height only
        # - later pooling layers: reduce both height and width
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # =================================================
            # BLOCK 1
            # =================================================
            # Input: [B, 1, H, W]
            # Output: [B, 8, H, W]
            #
            # A small 3x3 convolution detects simple features:
            # lines, edges, contrast boundaries, gate borders, etc.
            nn.Conv2d(
                in_channels=1,     # only Y channel
                out_channels=8,    # small number of filters for speed
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Pool height only, keep width unchanged.
            # This is important because width contains left/right info.
            #
            # Example:
            # H x W  ->  H/2 x W
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # =================================================
            # BLOCK 2
            # =================================================
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Now reduce both height and width.
            # At this point the network already has some useful features,
            # so shrinking width becomes less dangerous.
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # =================================================
            # BLOCK 3
            # =================================================
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # =================================================
            # BLOCK 4
            # =================================================
            # One extra convolution layer adds a bit more capacity,
            # but keeps the model still small enough to benchmark.
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------
        # This compresses the final feature map to 1x1 per channel.
        #
        # Why use this?
        # - much fewer parameters than a large fully connected layer
        # - faster
        # - simpler for embedded deployment
        #
        # After this layer:
        # [B, 32, h, w] -> [B, 32, 1, 1]
        # ----------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ----------------------------------------------------
        # REGRESSION HEAD
        # ----------------------------------------------------
        # Convert 32 learned features into one output in [0,1].
        #
        # We use Sigmoid because the target is normalized:
        #   left  = 0
        #   right = 1
        # ----------------------------------------------------
        self.head = nn.Sequential(
            nn.Flatten(),        # [B, 32, 1, 1] -> [B, 32]
            nn.Linear(32, 16),   # small hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),    # one output
            nn.Sigmoid()         # constrain output to [0,1]
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Input:
            x: tensor of shape [B, 1, H, W]

        Output:
            tensor of shape [B, 1]
            each value is in [0,1]
        """

        # Extract visual features
        x = self.features(x)

        # Compress spatial dimensions
        x = self.global_pool(x)

        # Predict normalized horizontal direction
        x = self.head(x)

        return x