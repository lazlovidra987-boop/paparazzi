import os
import sys

sys.path.append(os.path.dirname(__file__))

import torch
from model import DirectionCNN


def run_inference(height=240, width=520, device="cpu"):
    """
    Run one dummy inference through the model.
    """

    model = DirectionCNN().to(device)
    model.eval()

    x = torch.randn(1, 1, height, width, device=device)

    with torch.no_grad():
        y = model(x)

    print("======================================")
    print("Inference result")
    print("======================================")
    print(f"Input shape  : {tuple(x.shape)}")
    print(f"Output shape : {tuple(y.shape)}")
    print(f"Output value : {y.item():.6f}")
    print("======================================")


if __name__ == "__main__":
    run_inference(height=240, width=520, device="cpu")