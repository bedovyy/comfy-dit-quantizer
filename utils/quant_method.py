import torch
from comfy_kitchen.float_utils import (
    F8_E4M3_MAX,
    F4_E2M1_MAX,
    _f32_to_floatx_unpacked,
    _float8_round,
    pack_uint4,
    roundup,
    to_blocked,
)

_E2M1_POSITIVE_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    epsilon: float = 0.0,
    pad_16x: bool = False,
    use_mse_4_6: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))
            # Note: We update orig_shape because the output tensor logic below assumes x.shape matches
            # what we want to produce. If we pad here, we want the padded output.
            orig_shape = x.shape

    block_size = 16

    x = x.reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x), dim=-1)

    if use_mse_4_6:
        candidate_scales_6 = max_abs.to(torch.float32) / 6.0
        candidate_scales_4 = max_abs.to(torch.float32) / 4.0

        def e2m1_round(x: torch.Tensor) -> torch.Tensor:
            lut = _E2M1_POSITIVE_VALUES.to(x.device, x.dtype)
            sign = torch.sign(x)
            abs_x = x.abs()
            diffs = (abs_x.unsqueeze(-1) - lut).abs()
            idx = diffs.argmin(dim=-1)
            return torch.where(x == 0, torch.zeros_like(x), sign * lut[idx])

        def compute_mse_error(orig_x, scale_val, v_max, p_scale):
            s_block_fp8 = torch.clamp(scale_val / p_scale, max=F8_E4M3_MAX)
            s_block_fp32 = _float8_round(s_block_fp8)
            actual_scale = (p_scale * s_block_fp32).unsqueeze(-1)
            safe_scale = torch.where(actual_scale == 0, torch.ones_like(actual_scale), actual_scale)

            q = (orig_x / safe_scale).clamp(-v_max, v_max)
            q_r  = e2m1_round(q)
            dq = q_r * safe_scale

            return torch.mean((orig_x - dq)**2, dim=-1)

        mse_6 = compute_mse_error(x, candidate_scales_6, 6.0, per_tensor_scale)
        mse_4 = compute_mse_error(x, candidate_scales_4, 4.0, per_tensor_scale)
        block_scale = torch.where(mse_4 < mse_6, candidate_scales_4, candidate_scales_6)
    else:
        block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX

    scaled_block_scales = block_scale / per_tensor_scale
    scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
    total_scale = per_tensor_scale * scaled_block_scales_fp32

    # Handle zero blocks (from padding): avoid 0/0 NaN
    zero_scale_mask = (total_scale == 0)
    total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)

    data_scaled = x.float() / total_scale_safe.unsqueeze(-1)
    data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)

    out_scales = scaled_block_scales_fp8

    if use_mse_4_6:
        v_max_dynamic = torch.where(mse_4 < mse_6, 4.0, 6.0).unsqueeze(-1)
    else:
        v_max_dynamic = 6.0
    data_scaled = torch.clamp(data_scaled, -v_max_dynamic, v_max_dynamic)

    data_scaled = data_scaled.view(orig_shape)

    data_lp = _f32_to_floatx_unpacked(data_scaled, 2, 1)
    data_lp = pack_uint4(data_lp)
    blocked_scales = to_blocked(out_scales.to(torch.float8_e4m3fn), flatten=False)
    return data_lp, blocked_scales

def quantize_per_tensor_int8(x, scale):
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def dequantize_per_tensor_int8(x, scale):
    return x.float() * scale

def quantize_rowwise_int8(x, scales):
    x_float = x.float()
    quantized = torch.zeros_like(x, dtype=torch.int8)

    for i in range(scales.numel()):
        row = x_float[i] * (1.0 / scales[i])
        quantized[i] = torch.clamp(row.round(), -128, 127).to(torch.int8)

    return quantized

def dequantize_rowwise_int8(x, scales):
    x_float = x.float()
    dequantized = torch.zeros_like(x_float)

    for i in range(scales.numel()):
        dequantized[i] = x_float[i] * scales[i]

    return dequantized
