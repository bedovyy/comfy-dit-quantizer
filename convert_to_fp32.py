import torch
import safetensors.torch
import os
import sys

def convert_bf16_safetensors_to_fp32(input_path, output_path):
    state_dict = safetensors.torch.load_file(input_path)
    new_state_dict = {}
    for key, tensor in state_dict.items():
        print(f"{key} ({tensor.dtype}) -> torch.float32")
        new_tensor = tensor.to(torch.float32)
        new_state_dict[key] = new_tensor
    safetensors.torch.save_file(new_state_dict, output_path)
    print(f"output_path: {output_path}")

if __name__ == "__main__":
    assert len(sys.argv) == 3, f"usage: {sys.argv[0]} SOURCE TARGET"
    input_path, output_path = sys.argv[1:3]
    convert_bf16_safetensors_to_fp32(input_path, output_path)
