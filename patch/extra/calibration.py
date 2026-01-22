import atexit
import json
import os
import torch
from typing import Optional, Dict

CALIB_AMAX: Dict[str, torch.Tensor] = {}
ENABLED = os.getenv("COMFY_CALIB", "0") == "1"
VALID_FORMATS = {"nvfp4", "float8_e4m3fn", "float8_e5m2"}

def record_amax_from_tensor(layer_name, quant_format, x):
    if not ENABLED or quant_format not in VALID_FORMATS or not layer_name:
        return
    with torch.no_grad():
        x_fp32 = x.float()
        if torch.isinf(x_fp32).any() or torch.isnan(x_fp32).any():
            print(f"[CALIB_WARN] {layer_name}: Inf/NaN after fp32, skipping")
            return

        amax = x.detach().abs().amax()
        prev = CALIB_AMAX.get(layer_name)
        if prev is None:
            CALIB_AMAX[layer_name] = amax
        else:
            if prev.device != amax.device:
                prev = prev.to(device=amax.device)
            CALIB_AMAX[layer_name] = torch.maximum(prev, amax)

def dump(path: Optional[str] = None, clear: bool = False):
    if not ENABLED:
        return
    if path is None:
        path = os.path.join(os.getcwd(), "CALIB_AMAX.json")
    items = {k: float(v.detach().cpu().item()) for k, v in sorted(CALIB_AMAX.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[CALIB_AMAX_DUMP] {len(items)} layers -> {path}")
    if clear:
        CALIB_AMAX.clear()

if ENABLED:
    atexit.register(dump)
