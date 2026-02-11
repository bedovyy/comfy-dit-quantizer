import argparse
import json
import os
import torch
import comfy_kitchen as ck
from comfy_kitchen.float_utils import F8_E4M3_MAX, F4_E2M1_MAX
from safetensors.torch import load_file, save_file
import utils
from utils import print_layer_metrics, print_layer_header

QUANTIZABLE_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
ALLOWED_QTYPES = {"float8_e4m3fn", "nvfp4", "mxfp8", "int8_tensorwise"}

device = utils.get_device()


def parse_args():
    p = argparse.ArgumentParser(
        prog="quantize.py",
        description="Quantize safetensors weights with rule-based policies.",
    )
    p.add_argument("json", help="Quant config JSON path")
    p.add_argument("src", nargs="*", help="Source safetensors path")
    p.add_argument("dst", help="Target safetensors path")
    p.add_argument("-d", "--downcast-fp32", choices=("fp16", "bf16"), default=None, metavar="{fp16,bf16}",
                   help="Cast fp32 tensors to the selected dtype (default: keep FP32).")
    p.add_argument("-m", "--method", choices=("amax", "mse"), default="mse", metavar="{amax, mse}",
                   help="Set calibration method (default: mse).")
    p.add_argument("-n", "--n-samples", default=None, type=int, help="num of samples for calibration method")
    p.add_argument("-q", "--quiet", action="store_true", help="no verbose.")
    p.add_argument("-t", "--test", action="store_true", help="does not save output")
    return p.parse_args()

def quantize_weight(weight, key, quantized_state_dict, quantization_layers, qtype, qformat, method, n_samples, verbose=True):
    layer_name = key[:-7]
    
    if qtype == "nvfp4":
        if method == "mse":
            weight_scale_2 = utils.scale_mse_nvfp4(weight, n_samples=n_samples)
        else:
            weight_scale_2 = utils.scale_amax_nvfp4(weight)
        with ck.use_backend("triton"): # triton supports conversion from fp32
            weight_quantized, weight_scale = ck.quantize_nvfp4(weight, weight_scale_2)
        if verbose: print_layer_metrics(layer_name, weight, weight_quantized, weight_scale_2, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale_2"] = weight_scale_2.cpu()
    elif qtype == "mxfp8":
        orig_dtype = weight.dtype
        orig_shape = tuple(weight.shape)
        weight_quantized, weight_scale = ck.quantize_mxfp8(weight)
        if verbose: print(layer_name) # impl. later
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
    elif qtype == "int8_tensorwise":
        if method == "mse":
          weight_scale = utils.scale_mse_int8(weight, n_samples=n_samples)
        else:
          weight_scale = utils.scale_amax_int8(weight)
        weight_quantized = utils.quantize_per_tensor_int8(weight, weight_scale)
        if verbose: print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
    else: # fp8
        if method == "mse":
            weight_scale = utils.scale_mse_fp8(weight, n_samples=n_samples)
        else:
            weight_scale = utils.scale_amax_fp8(weight)
        weight_quantized = ck.quantize_per_tensor_fp8(weight, weight_scale)
        if verbose: print_layer_metrics(layer_name, weight, weight_quantized, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()

    if qtype == "mxfp8":
        quant_info = { "format": qtype, "orig_dtype": "torch.bfloat16", "orig_shape": orig_shape }
    else:
        quant_info = { "format": qtype }

    if qformat == "comfy_quant":
        quantized_state_dict[f"{layer_name}.comfy_quant"] = torch.tensor(
                list(json.dumps(quant_info).encode("utf-8")), dtype=torch.uint8)
    else: # 1.0
        quantization_layers[layer_name] = quant_info

def store_with_optional_downcast(tensor, key, quantized_state_dict, cast_to, verbose=True):
    if tensor.dtype == torch.float32 and cast_to != None:
        casted_weight = tensor.to(dtype=cast_to)
        quantized_state_dict[key] = casted_weight.cpu()

        if verbose and key.endswith(".weight"):
            layer_name = key[:-7]
            print_layer_metrics(layer_name, tensor, casted_weight)
    else:
        quantized_state_dict[key] = tensor.cpu()

def first_matching_qtype_for_key(key, rules):
    for r in rules:
        if any(p in key for p in r.get("match", [])):
            qtype = r.get("policy")
            return qtype if qtype in ALLOWED_QTYPES else None
    return None

def main():
    args = parse_args()
    cast_to = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.downcast_fp32, None)
    assert ".json" in args.json, f"{args.json} is not .json file."
    with open(args.json, "r", encoding="utf-8") as f:
        config = json.load(f)
    qformat = config.get("format", "1.0")
    block_names = config.get("block_names", ["block", "transformer", "layer"])
    rules = config.get("rules", [])
    
    quantized_state_dict, quantization_layers = {}, {}
    for f in args.src:
        print(f)
        state_dict = load_file(f)

        if not args.quiet: print_layer_header()
    
        for key, tensor in state_dict.items():
            if not (any(b in key for b in block_names) and key.endswith(".weight")
                    and tensor.dtype in QUANTIZABLE_WEIGHT_DTYPES and tensor.ndim == 2):
                store_with_optional_downcast(tensor, key, quantized_state_dict, cast_to, verbose=not args.quiet)
                continue

            qtype = first_matching_qtype_for_key(key, rules)
            if qtype is None:
                store_with_optional_downcast(tensor, key, quantized_state_dict, cast_to, verbose=not args.quiet)
            else:
                quantize_weight(tensor.to(device), key, quantized_state_dict, quantization_layers, qtype, qformat, args.method, args.n_samples, verbose=not args.quiet)

    if not args.test:
        metadata = (
            {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": quantization_layers})}
            if qformat != "comfy_quant" else None
        )
        save_file(quantized_state_dict, args.dst, metadata=metadata)
        total_bytes = os.path.getsize(args.dst)
        print(f"Output: {args.dst} ({round(total_bytes / (1024**3), 2)}GB)")

if __name__ == "__main__":
    main()
