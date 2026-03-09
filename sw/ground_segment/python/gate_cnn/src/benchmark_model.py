# benchmark_model.py
# ------------------------------------------------------------
# Simple benchmark script for DirectionCNN
#
# This script does NOT test accuracy.
# It only measures inference speed.
#
# You can later compare multiple architectures this way.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(__file__))

import time
import torch
from model import DirectionCNN


def benchmark_model(
    height=240,
    width=520,
    batch_size=1,
    num_warmup=20,
    num_runs=200,
    device="cpu"
):
    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------
    model = DirectionCNN().to(device)
    model.eval()

    # --------------------------------------------------------
    # Create dummy input
    # Shape = [batch, channels, height, width]
    # We use 1 channel because we assume Y-only input.
    # --------------------------------------------------------
    x = torch.randn(batch_size, 1, height, width, device=device)

    # --------------------------------------------------------
    # Warmup runs
    # --------------------------------------------------------
    # Warmup is useful because first runs are often slower.
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time

    print("======================================")
    print("Benchmark results")
    print("======================================")
    print(f"Input size        : {height} x {width}")
    print(f"Batch size        : {batch_size}")
    print(f"Device            : {device}")
    print(f"Runs              : {num_runs}")
    print(f"Average time      : {avg_time * 1000:.3f} ms")
    print(f"Approx. FPS       : {fps:.2f}")
    print("======================================")


if __name__ == "__main__":
    benchmark_model(
        height=240,
        width=520,
        batch_size=1,
        num_warmup=20,
        num_runs=200,
        device="cpu"
    )