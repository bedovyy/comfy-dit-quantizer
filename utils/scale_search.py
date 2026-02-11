import torch
import comfy_kitchen as ck
from comfy_kitchen.float_utils import F8_E4M3_MAX, F4_E2M1_MAX
import utils

#NUM_SAMPLE_DEFAULT = 131_072
NUM_SAMPLE_DEFAULT = None

INT8_MAX = 127.0

def sample_flat(w, n, include_absmax=True):
    if w.numel() == 0:
        return w.new_empty((0,), dtype=torch.float32)

    x = w.flatten()
    if x.numel() <= n:
        return x.float()

    k = max(0, n - (1 if include_absmax else 0))
    idx = torch.randint(x.numel(), (k,), device=x.device)
    if include_absmax:
        idx = torch.cat([idx, x.abs().argmax().view(1)])
    return x[idx].float()

def sample_block16(w, n, include_absmax=True):
    if w.numel() == 0:
        return w.new_empty((0, 16), dtype=torch.float32)

    w = w.contiguous()
    r, c = w.shape
    assert r % 16 == 0 and c % 16 == 0, f"Rows and columns must be divisible by 16, got shape ({r}, {c})"
    x16 = w.view(r, c // 16, 16)

    n = ((n + 255) // 256) * 256
    n_blocks = n // 16
    total_blocks = x16.shape[0] * x16.shape[1]
    if total_blocks <= n_blocks:
        return x16.reshape(-1, 16).float()

    k = n_blocks - (1 if include_absmax else 0)
    ridx = torch.randint(x16.shape[0], (k,), device=w.device)
    bidx = torch.randint(x16.shape[1], (k,), device=w.device)

    if include_absmax:
        flat_pos = w.abs().view(-1).argmax().item()
        ar, ac = divmod(flat_pos, c)
        ridx = torch.cat([ridx, w.new_tensor([ar], dtype=torch.long)])
        bidx = torch.cat([bidx, w.new_tensor([ac // 16], dtype=torch.long)])

    return x16[ridx, bidx].float()


def scale_mse_nvfp4(w, n_samples=NUM_SAMPLE_DEFAULT, ratios=(0.8 + 0.05*i for i in range(13))):
    x = sample_block16(w, n_samples) if n_samples != None else w.float()
    if x.numel() == 0 :
        return w.new_tensor(0.0, dtype=torch.float32)

    amax = torch.amax(x.abs())
    if amax.item() == 0.0:
        return w.new_tensor(0.0, dtype=torch.float32)

    base = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
    best_scale, best_mse = base, float("inf")

    for r in ratios:
        scale = base * r
        with ck.use_backend("triton"):
            quant, block_scales = ck.quantize_nvfp4(x, scale)
            dequant = ck.dequantize_nvfp4(quant, scale, block_scales, output_type=torch.float32)
        mse = (x - dequant).pow(2).mean().item()
        if mse < best_mse:
            best_scale, best_mse = scale, mse

    return best_scale.to(dtype=torch.float32)

def scale_mse_fp8(w, n_samples=NUM_SAMPLE_DEFAULT, ratios=(0.85 + 0.025*i for i in range(13))):
    x = sample_flat(w, n_samples) if n_samples != None else w.float()
    if x.numel() == 0:
        return w.new_tensor(0.0, dtype=torch.float32)

    amax = torch.amax(x.abs())
    if amax.item() == 0.0:
        return w.new_tensor(0.0, dtype=torch.float32)

    base = amax / F8_E4M3_MAX
    best_scale, best_mse = base, float("inf")

    for r in ratios:
        scale = base * r
        quant = ck.quantize_per_tensor_fp8(x, scale, output_type=torch.float8_e4m3fn)
        dequant = ck.dequantize_per_tensor_fp8(quant, scale, output_type=torch.float32)
        mse = (x - dequant).pow(2).mean().item()
        if mse < best_mse:
            best_scale, best_mse = scale, mse

    return best_scale.to(dtype=torch.float32)

# TODO: Refactoring
def scale_mse_int8(w, n_samples=NUM_SAMPLE_DEFAULT, ratios=(0.9 + 0.01*i for i in range(21))):
    x = sample_flat(w, n_samples) if n_samples is not None else w.float()
    if x.numel() == 0:
        return w.new_tensor(0.0, dtype=torch.float32)

    amax = torch.amax(x.abs())
    if amax.item() == 0.0:
        return w.new_tensor(0.0, dtype=torch.float32)

    base = amax / INT8_MAX
    best_scale, best_mse = base, float("inf")

    for r in ratios:
        scale = base * r
        quant = utils.quantize_per_tensor_int8(x, scale)
        dequant = utils.dequantize_per_tensor_int8(quant, scale)
        mse = (x - dequant).pow(2).mean().item()

        if mse < best_mse:
            best_scale, best_mse = scale, mse

    return best_scale.to(dtype=torch.float32)

def scale_amax_nvfp4(w):
    return torch.amax(w.abs()).to(dtype=torch.float32) / (F8_E4M3_MAX * F4_E2M1_MAX)

def scale_amax_fp8(w):
    return torch.amax(w.abs()).to(dtype=torch.float32) / F8_E4M3_MAX

def scale_amax_int8(w):
    return torch.amax(w.abs()).to(dtype=torch.float32) / INT8_MAX
