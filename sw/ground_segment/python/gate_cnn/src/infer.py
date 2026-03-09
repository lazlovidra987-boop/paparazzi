import os
import torch
from model import DirectionCNN


def write_array(f, name, tensor):
    data = tensor.detach().cpu().numpy().flatten()
    shape = list(tensor.shape)

    f.write(f"// shape: {shape}\n")
    f.write(f"static const float {name}[] = {{\n")

    for i, val in enumerate(data):
        f.write(f"    {float(val):.8f}f")
        if i != len(data) - 1:
            f.write(",")
        if (i + 1) % 8 == 0:
            f.write("\n")
        else:
            f.write(" ")

    f.write("\n};\n\n")


def main():
    model = DirectionCNN()

    # Kies hier waar je de C header wilt hebben
    output_path = r"c:/Users/Mikes/paparazzi/sw/airborne/modules/gate_cnn/model_weights.h"

    state = model.state_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        f.write("/* CNN MODULE: auto-generated model weights header */\n\n")

        for name, tensor in state.items():
            c_name = name.replace(".", "_")
            write_array(f, c_name, tensor)

        f.write("#endif\n")

    print("======================================")
    print("Export finished")
    print("======================================")
    print(f"Saved weights to:\n{output_path}")
    print("======================================")


if __name__ == "__main__":
    main()