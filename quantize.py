import argparse
import json
import torch
import comfy_kitchen as ck
from safetensors.torch import load_file, save_file

QUANTIZABLE_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
ALLOWED_QTYPES = {"float8_e4m3fn", "nvfp4"}

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.xpu.is_available():  # Intel GPU
    device = torch.device('xpu')


def parse_args():
    p = argparse.ArgumentParser(
        prog="quantize.py",
        description="Quantize safetensors weights with rule-based policies.",
    )
    p.add_argument("json", help="Quant config JSON path")
    p.add_argument("src", help="Source safetensors path")
    p.add_argument("dst", help="Target safetensors path")
    return p.parse_args()

def get_metrics(original, quantized, global_scale, block_scales=None):
    if block_scales is not None and quantized.dtype == torch.uint8:
        dequantized = ck.dequantize_nvfp4(quantized, global_scale, block_scales, original.dtype) # nvfp4
    else: 
        dequantized = ck.dequantize_per_tensor_fp8(quantized, global_scale, original.dtype) # fp8
    mse = torch.mean((original - dequantized)**2)
    mae = torch.mean(torch.abs(original - dequantized))
    max_err = torch.max(torch.abs(original - dequantized))  # Max Error
    rel_max_err = (max_err / (original.abs().amax() + 1e-8)).item() * 100  # Relative (%)
    return mse, mae, max_err, rel_max_err

def quantize_layer(tensor, key, quantized_state_dict, quantization_layers, qtype, qformat):
    layer_name = key.replace(".weight", "")
    amax = torch.amax(tensor.abs()).to(torch.float32)
    
    if qtype == "nvfp4":
        weight_scale_2 = amax / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)
        with ck.use_backend("triton"): # triton supports conversion from fp32
            weight_quantized, weight_scale = ck.quantize_nvfp4(tensor, weight_scale_2, epsilon=1e-6)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale_2"] = weight_scale_2.cpu()
    else: # fp8
        weight_scale = amax / ck.float_utils.F8_E4M3_MAX
        weight_quantized = ck.quantize_per_tensor_fp8(tensor, weight_scale)
        quantized_state_dict[key] = weight_quantized.cpu()
        quantized_state_dict[f"{layer_name}.weight_scale"] = weight_scale.cpu()

    if qformat == "comfy_quant":
        quantized_state_dict[f"{layer_name}.comfy_quant"] = torch.tensor(
                list(json.dumps({"format": qtype}).encode("utf-8")), dtype=torch.uint8)
    else: # 1.0
        quantization_layers[layer_name] = {"format": qtype}

    if qtype == "nvfp4":
        fmt = "nvfp4"
        mse, mae, max_err, rel_max_err = get_metrics(tensor, weight_quantized, weight_scale_2, weight_scale)
        tail = f"global_scale={weight_scale_2:.8f} mse={mse:.6f} mae:{mae:.6f} max_err:{max_err:.6f} rel:{rel_max_err:.2f}"
    else:  # fp8
        fmt = "fp8"
        mse, mae, max_err, rel_max_err = get_metrics(tensor, weight_quantized, weight_scale)
        tail = f"global_scale={weight_scale:.8f} mse={mse:.6f} mae:{mae:.6f} max_err:{max_err:.6f} rel:{rel_max_err:.2f}"
    print(f"[{layer_name.partition('.')[2][:20]:<20} {fmt:>5}] amax:{amax.item():.4f}", tail)

def first_matching_qtype_for_key(key, rules):
    for r in rules:
        if any(p in key for p in r.get("match", [])):
            qtype = r.get("policy")
            return qtype if qtype in ALLOWED_QTYPES else None
    return None



def main():
    args = parse_args()
    with open(args.json, "r", encoding="utf-8") as f:
        config = json.load(f)
    qformat = config.get("format", "1.0")
    block_name = config.get("block_name", "block")
    rules = config.get("rules", [])
    cast_to = torch.bfloat16 # temp
    
    state_dict = load_file(args.src)
    quantized_state_dict, quantization_layers = {}, {}
    
    for key, tensor in state_dict.items():
        if not (block_name in key and key.endswith(".weight") and tensor.dtype in QUANTIZABLE_WEIGHT_DTYPES and tensor.ndim == 2):
            quantized_state_dict[key] = tensor.to(cast_to)
#            quantized_state_dict[key] = tensor
            continue
        
        qtype = first_matching_qtype_for_key(key, rules)
        if qtype is None:
            quantized_state_dict[key] = tensor.to(cast_to)
#            quantized_state_dict[key] = tensor
        else:
            quantize_layer(tensor.to(device), key, quantized_state_dict, quantization_layers, qtype, qformat)

    
    metadata = (
        {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": quantization_layers})}
        if qformat != "comfy_quant" else None
    )
    save_file(quantized_state_dict, args.dst, metadata=metadata)

if __name__ == "__main__":
    main()
