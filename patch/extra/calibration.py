import atexit
import json
import os
import torch
import threading
import math

from typing import Optional, Dict

thread_local = threading.local()
MODEL_SIGMA_RANGE = {}  # {model_id: (sigma_min, sigma_max)}
CALIB_DATA: Dict[str, Dict[str, float]] = {}
ENABLED = os.getenv("COMFY_CALIB", "0") == "1"
VALID_FORMATS = {"nvfp4", "float8_e4m3fn", "float8_e5m2"}

def set_timestep(timestep, model_sampling=None):
    if isinstance(timestep, torch.Tensor):
        timestep = float(timestep[0]) if timestep.numel() > 0 else None
    thread_local.timestep = timestep

    if model_sampling is not None:
        model_id = id(model_sampling)
        if model_id not in MODEL_SIGMA_RANGE:
            MODEL_SIGMA_RANGE[model_id] = (
                float(model_sampling.sigma_min),
                float(model_sampling.sigma_max)
            )

def get_timestep_bin(timestep, sigma_min, sigma_max, num_bins=6):
    if timestep is None or timestep <= 0:
        return 0
    if isinstance(timestep, torch.Tensor):
        timestep = float(timestep.item())

    normalized = (timestep - sigma_min) / (sigma_max - sigma_min + 1e-8)
    bin_idx = bin_idx = int((1.0 - normalized) * num_bins)
    return min(max(bin_idx, 0), num_bins-1)

def _compute_safe_stats(abs_x: torch.Tensor, sample_size=131_072) -> Dict[str, float]:
    numel = abs_x.numel()
    amax = float(torch.amax(abs_x).item())

    if numel <= 524_288:
        return {
            "amax": amax,
            "p99_9": float(torch.quantile(abs_x, 0.999).item()),
            "std": float(abs_x.std().item()),
            "min": float(abs_x.min().item()),
            "max": float(abs_x.max().item()),
            "count": 1
        }

    flat_x = abs_x.flatten()

    indices = torch.randperm(numel, device=abs_x.device)[:sample_size]
    sample = flat_x[indices]

    return {
        "amax": amax,
        "p99_9": float(torch.quantile(sample, 0.999).item()),
        "std": float(sample.std().item()),
        "min": float(sample.min().item()),
        "max": float(sample.max().item()),
        "count": 1,
    }

def record_amax_from_tensor(layer_name, quant_format, x):
    if not ENABLED or quant_format not in VALID_FORMATS or not layer_name:
        return

    timestep = getattr(thread_local, "timestep", None)

    if timestep is None:
        return
    if not MODEL_SIGMA_RANGE:
        print(f"[CALIB_WARN] No sigma range recorded yet, skipping {layer_name}")
        return

    sigma_min, sigma_max = list(MODEL_SIGMA_RANGE.values())[-1]
    bin_idx = get_timestep_bin(timestep, sigma_min, sigma_max)
    key = f"{layer_name}_bin{bin_idx}"

    with torch.no_grad():
        x_fp32 = x.float()
        if torch.isinf(x_fp32).any() or torch.isnan(x_fp32).any():
            print(f"[CALIB_WARN] {layer_name}: Inf/NaN after fp32, skipping")
            return

        abs_x = x_fp32.detach().abs()
        stats = _compute_safe_stats(abs_x)

        prev_stats = CALIB_DATA.get(key)
        if prev_stats is not None:
            for k in stats:
                if k in prev_stats:
                    if k == "count":
                        stats[k] = prev_stats[k] + stats[k]
                    elif k == "min":
                        stats[k] = min(stats[k], prev_stats[k])
                    else:
                        stats[k] = max(stats[k], prev_stats[k])
        CALIB_DATA[key] = stats

def dump(path: Optional[str] = None, clear: bool = False):
    if not ENABLED:
        return
    if path is None:
        path = os.path.join(os.getcwd(), "CALIB_DATA.json")

    output = {
        "layers": CALIB_DATA,
        "sigma_ranges": {
            str(k): {"min": v[0], "max": v[1]}
            for k, v in MODEL_SIGMA_RANGE.items()
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_bins = len(CALIB_DATA)
    unique_layers = len({k.rsplit('_bin', 1)[0] for k in CALIB_DATA})
    print(f"[CALIB_DATA_DUMP] {total_bins} layer-bins ({unique_layers} unique layers) -> {path}")
    if clear:
        CALIB_DATA.clear()
        MODEL_SIGMA_RANGE.clear()

if ENABLED:
    atexit.register(dump)
