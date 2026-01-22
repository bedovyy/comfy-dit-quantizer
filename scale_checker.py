import json
import sys
from safetensors import safe_open

target_keywords = ['input_scale', 'scale_input', 'weight_scale_2', 'weight_scale', 'scale_weight', 'comfy_quant']

with safe_open(sys.argv[1], framework="pt") as f:
    metadata = f.metadata()
    print("=== Metadata ===")
    if metadata:
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("no metadata.")

    print("=== layers ===")
    for key in f.keys():
        tensor_slice = f.get_slice(key)
        tensor = f.get_tensor(key)
        print(f"{key}: {tensor_slice.get_shape()}: {tensor_slice.get_dtype()}", end=" ")
        if any(word in key for word in target_keywords) and tensor.numel() == 1:
            print(f"{tensor.item():.12f}")
        elif tensor.dim() == 1 and 'comfy_quant' in key:
            print(json.loads(bytes(tensor.detach().cpu().view(-1).tolist()).decode("utf-8")))
        else:
            print()
