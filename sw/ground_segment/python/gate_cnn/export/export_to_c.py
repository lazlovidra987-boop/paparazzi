
import os
import sys
import argparse
import torch
# model.py lives in ../src relative to this file
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))

# Assuming your model definition is in a file named model_dronet.py
# If it's in the same file, you can just import it directly.
from model_dronet import DirectionGateNet

def write_array(f, name: str, tensor):
    """Write one float array to the header file."""
    data = tensor.detach().cpu().numpy().flatten()
    shape = list(tensor.shape)

    # Clean the name: replace dots with underscores
    c_name = name.replace(".", "_")
    
    # Remove common prefixes if they exist (e.g., 'module.' from DataParallel)
    if c_name.startswith("module_"):
        c_name = c_name[7:]

    f.write(f"// shape: {shape}  elements: {len(data)}\n")
    f.write(f"static const float {c_name}[] = {{\n")

    for i, val in enumerate(data):
        f.write(f"    {float(val):.8f}f")
        if i != len(data) - 1:
            f.write(",")
        if (i + 1) % 8 == 0 or i == len(data) - 1:
            f.write("\n")
        else:
            f.write(" ")
    f.write("};\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth file")
    parser.add_argument("--output", default="model_weights.h", help="Output C header")
    args = parser.parse_args()

    model = DirectionGateNet(output_mode="probability")
    
    try:
        # Load state dict
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        
        # If the checkpoint is a dict containing 'state_dict', extract it
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print(f"Successfully loaded: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    state = model.state_dict()

    # Ensure output directory exists
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        f.write("/* Auto-generated weights for DirectionGateNet */\n\n")

        for name, tensor in state.items():
            write_array(f, name, tensor)

        f.write("#endif // MODEL_WEIGHTS_H\n")

    print(f"✅ Exported {len(state)} tensors to: {output_path}")
    
    # Print summary to console for verification
    print("\nVerifying C Array Names:")
    for name in state.keys():
        clean_name = name.replace(".", "_").replace("module_", "")
        print(f"  {clean_name}")

if __name__ == "__main__":
    main()