import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_MAX = 6.0  # E2M1 max (common FP4 range assumption)
CALIBRATION_MARGIN = 1.0

def _quant_format_from_dtype(dt) -> str:
    s = str(dt).upper()
    if dt == torch.uint8 or s in ("U8", "UINT8"):
        return "nvfp4"
#    if dt == torch.float8_e4m3fn or "E4M3" in s:
#        return "float8_e4m3fn"
#    if dt == torch.float8_e5m2 or "E5M2" in s:
#        return "float8_e5m2"
    return ""  # unsupported

def _input_scale_from_format(qfmt: str, value: float) -> float:
    value = value * CALIBRATION_MARGIN
    if qfmt == "nvfp4":
        fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
        return value / (fp8_max * FP4_MAX)

    if qfmt in ("float8_e4m3fn", "float8_e5m2"):
        fp8_dt = torch.float8_e4m3fn if qfmt == "float8_e4m3fn" else torch.float8_e5m2
        fp8_max = float(torch.finfo(fp8_dt).max)
        return value / fp8_max

    raise ValueError(f"unsupported quant format: {qfmt}")

def parse_detailed_format(layers, key):
    data = {}
    for layer_bin, stats in layers.items():
        if "_bin0" not in layer_bin:
            continue
        layer_name = layer_bin.replace("_bin0", "")
        data[layer_name] = stats[key]
    return data

def main():
    json_path, input_path, output_path = sys.argv[1:4]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "layers" in data:
        data =  parse_detailed_format(data["layers"], "amax")

    out_tensors = {}

    with safe_open(input_path, framework="pt") as f:
        metadata = dict(f.metadata() or {})


        for k in f.keys():
            out_tensors[k] = f.get_tensor(k)

        for layer_name, value in data.items():
#            layer_name = "net." + layer_name # anime-preview

            weight_key = f"{layer_name}.weight"
            
            dt = f.get_slice(weight_key).get_dtype()

            qfmt = _quant_format_from_dtype(dt)
            if not qfmt:
                continue

            input_scale = _input_scale_from_format(qfmt, float(value))
            out_tensors[f"{layer_name}.input_scale"] = torch.tensor(input_scale, dtype=torch.float32)

            scale_key = f"{layer_name}.weight_scale_2" if qfmt == "nvfp4" else f"{layer_name}.weight_scale"
            weight_scale = out_tensors[scale_key].item()
            print(f"[{layer_name.partition('.')[2]:<25} {qfmt[:6]:>6}]",
                  f"weight_scale={weight_scale:.4}, input_scale={input_scale:.4} diff={input_scale / weight_scale:.4f}")

    save_file(out_tensors, output_path, metadata=metadata)

if __name__ == "__main__":
    main()
