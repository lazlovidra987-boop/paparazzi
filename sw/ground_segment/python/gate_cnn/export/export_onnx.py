import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import DirectionCNN


def export_onnx(
    output_path="/home/mike/paparazzi/sw/ground_segment/python/gate_cnn/export/direction_cnn.onnx",
    height=520,
    width=240,
):
    model = DirectionCNN()
    model.eval()

    dummy_input = torch.randn(1, 1, height, width)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print("======================================")
    print("ONNX export finished")
    print("======================================")
    print(f"Saved ONNX model to:\n{output_path}")
    print("======================================")


if __name__ == "__main__":
    export_onnx()