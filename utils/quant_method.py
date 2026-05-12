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
            orig_shape = x.shape

    block_size = 16

    x = x.reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x), dim=-1)

    if use_mse_4_6:
        lut_boundaries = ((_E2M1_POSITIVE_VALUES[:-1] + _E2M1_POSITIVE_VALUES[1:]) / 2)

        def e2m1_round(t: torch.Tensor) -> torch.Tensor:
            lut = _E2M1_POSITIVE_VALUES.to(t.device, t.dtype)
            boundaries = lut_boundaries.to(t.device, t.dtype)
            sign = torch.sign(t)
            idx = torch.bucketize(t.abs(), boundaries).clamp_(0, len(lut) - 1)
            sign.mul_(lut[idx])
            sign.masked_fill_(t == 0, 0.0)
            return sign

        def compute_mse_error(orig_x, scale_val, v_max, p_scale):
            s_block_fp8 = torch.clamp(scale_val / p_scale, max=F8_E4M3_MAX)
            s_block_fp32 = _float8_round(s_block_fp8)
            del s_block_fp8
            actual_scale = (p_scale * s_block_fp32).unsqueeze(-1)
            del s_block_fp32
            actual_scale.masked_fill_(actual_scale == 0, 1.0)

            q = (orig_x / actual_scale).clamp_(-v_max, v_max)
            q_r = e2m1_round(q)
            del q
            q_r.mul_(actual_scale)
            del actual_scale
            q_r.sub_(orig_x).pow_(2)
            return q_r.mean(dim=-1)

        def compute_mse_error_chunked(orig_x, scale_val, v_max, p_scale, chunk=256):
            results = []
            for i in range(0, orig_x.shape[0], chunk):
                r = compute_mse_error(
                    orig_x[i:i+chunk],
                    scale_val[i:i+chunk],
                    v_max,
                    p_scale,
                )
                results.append(r.cpu())
            return torch.cat(results, dim=0).to(orig_x.device)

        with torch.no_grad():
            candidate_scales_6 = max_abs.to(torch.float32) / 6.0
            candidate_scales_4 = max_abs.to(torch.float32) / 4.0
            del max_abs

            mse_6 = compute_mse_error_chunked(x, candidate_scales_6, 6.0, per_tensor_scale)
            mse_4 = compute_mse_error_chunked(x, candidate_scales_4, 4.0, per_tensor_scale)

            use_4_mask = (mse_4 < mse_6)
            block_scale = torch.where(use_4_mask, candidate_scales_4, candidate_scales_6)
            del mse_4, mse_6, candidate_scales_4, candidate_scales_6
            torch.cuda.empty_cache()
    else:
        block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX
        del max_abs

    scaled_block_scales = block_scale / per_tensor_scale
    del block_scale
    scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
    del scaled_block_scales
    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
    total_scale = per_tensor_scale * scaled_block_scales_fp32
    del scaled_block_scales_fp32, per_tensor_scale

    # Handle zero blocks (from padding): avoid 0/0 NaN
    zero_scale_mask = (total_scale == 0)
    total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)
    del total_scale

    data_scaled = x.to(torch.float32)
    del x
    torch.cuda.empty_cache()
    data_scaled /= total_scale_safe.unsqueeze(-1)
    del total_scale_safe

    zero_mask_expanded = zero_scale_mask.unsqueeze(-1)
    del zero_scale_mask
    data_scaled.masked_fill_(zero_mask_expanded, 0.0)
    del zero_mask_expanded

    out_scales = scaled_block_scales_fp8
    del scaled_block_scales_fp8

    if use_mse_4_6:
        v_max_dynamic = torch.where(
            use_4_mask.unsqueeze(-1),
            torch.tensor(4.0, dtype=data_scaled.dtype, device=data_scaled.device),
            torch.tensor(6.0, dtype=data_scaled.dtype, device=data_scaled.device),
        )
        del use_4_mask
        data_scaled.clamp_(-v_max_dynamic, v_max_dynamic)
        del v_max_dynamic
    else:
        data_scaled.clamp_(-6.0, 6.0)

    data_scaled = data_scaled.view(orig_shape)

    rows = data_scaled.shape[0]
    chunk = 128
    first_lp = pack_uint4(_f32_to_floatx_unpacked(data_scaled[:1], 2, 1))
    data_lp = torch.empty(rows, *first_lp.shape[1:], dtype=first_lp.dtype, device=data_scaled.device)
    del first_lp
    for i in range(0, rows, chunk):
        lp_chunk = _f32_to_floatx_unpacked(data_scaled[i:i+chunk], 2, 1)
        data_lp[i:i+chunk] = pack_uint4(lp_chunk)
        del lp_chunk
    del data_scaled
    torch.cuda.empty_cache()

    blocked_scales = to_blocked(out_scales.to(torch.float8_e4m3fn), flatten=False)
    del out_scales
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
