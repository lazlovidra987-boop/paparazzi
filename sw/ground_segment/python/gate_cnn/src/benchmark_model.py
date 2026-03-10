import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

import torch
from model import DirectionCNN

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def benchmark_pytorch(
    height=240,
    width=520,
    batch_size=1,
    num_warmup=20,
    num_runs=200,
    device="cpu"
):
    model = DirectionCNN().to(device)
    model.eval()

    x = torch.randn(batch_size, 1, height, width, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time

    print("======================================")
    print("PyTorch benchmark results")
    print("======================================")
    print(f"Input size        : {height} x {width}")
    print(f"Batch size        : {batch_size}")
    print(f"Device            : {device}")
    print(f"Runs              : {num_runs}")
    print(f"Average time      : {avg_time * 1000:.3f} ms")
    print(f"Approx. FPS       : {fps:.2f}")
    print("======================================")


def benchmark_onnx(
    onnx_path,
    height=240,
    width=520,
    batch_size=1,
    num_warmup=20,
    num_runs=200
):
    if not ONNX_AVAILABLE:
        print("onnxruntime is not installed.")
        return

    import numpy as np

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    x = np.random.randn(batch_size, 1, height, width).astype("float32")

    for _ in range(num_warmup):
        _ = session.run(None, {input_name: x})

    start = time.perf_counter()

    for _ in range(num_runs):
        _ = session.run(None, {input_name: x})

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time

    print("======================================")
    print("ONNX Runtime benchmark results")
    print("======================================")
    print(f"ONNX file         : {onnx_path}")
    print(f"Input size        : {height} x {width}")
    print(f"Batch size        : {batch_size}")
    print(f"Runs              : {num_runs}")
    print(f"Average time      : {avg_time * 1000:.3f} ms")
    print(f"Approx. FPS       : {fps:.2f}")
    print("======================================")


if __name__ == "__main__":
    benchmark_pytorch(
        height=240,
        width=520,
        batch_size=1,
        num_warmup=20,
        num_runs=200,
        device="cpu"
    )

    benchmark_onnx(
        onnx_path=r"c:/Users/Mikes/paparazzi/sw/ground_segment/python/gate_cnn/export/direction_cnn.onnx",
        height=240,
        width=520,
        batch_size=1,
        num_warmup=20,
        num_runs=200
    )