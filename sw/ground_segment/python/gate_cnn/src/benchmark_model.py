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
    height=120,
    width=160,
    batch_size=1,
    num_warmup=20,
    num_runs=200,
    device="cpu"
):
    model = DirectionCNN().to(device)
    model.eval()

    x = torch.randn(batch_size, 1, height, width, device=device)

    # Warmup — model now returns (heading, confidence) tuple
    with torch.no_grad():
        for _ in range(num_warmup):
            _heading, _conf = model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _heading, _conf = model(x)

    end = time.perf_counter()

    avg_time = (end - start) / num_runs
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
    print(f"<100ms target     : {'PASS' if avg_time < 0.1 else 'FAIL'}")
    print("======================================")


def benchmark_onnx(
    onnx_path,
    height=120,
    width=160,
    batch_size=1,
    num_warmup=20,
    num_runs=200
):
    if not ONNX_AVAILABLE:
        print("onnxruntime not installed — skipping ONNX benchmark.")
        print("Install with: pip install onnxruntime")
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

    avg_time = (end - start) / num_runs
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
    print(f"<100ms target     : {'PASS' if avg_time < 0.1 else 'FAIL'}")
    print("======================================")


if __name__ == "__main__":
    # --- PyTorch benchmark (run this on your laptop AND on the Bebop) ---
    benchmark_pytorch(
        height=120,
        width=160,
        batch_size=1,
        num_warmup=20,
        num_runs=200,
        device="cpu"        # Bebop has no GPU — always benchmark on CPU
    )

    # --- ONNX benchmark (optional, update path to your exported file) ---
    onnx_path = os.path.join(os.path.dirname(__file__), "direction_cnn.onnx")
    if os.path.exists(onnx_path):
        benchmark_onnx(
            onnx_path=onnx_path,
            height=120,
            width=160,
            batch_size=1,
            num_warmup=20,
            num_runs=200,
        )
    else:
        print(f"ONNX file not found at {onnx_path}, skipping ONNX benchmark.")
        print("Run export_onnx.py first to generate it.")