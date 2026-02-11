import torch
import comfy_kitchen as ck
import utils

def fixed_e(x, e=6, prec=4): return f"{x * (10**e):.{prec}f}e-{e}"

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')

def get_metrics(original, quantized, global_scale=None, block_scales=None):
    if block_scales is not None and quantized.dtype == torch.uint8:
        assert global_scale is not None, "nvfp4 requires global_scale"
        dequantized = ck.dequantize_nvfp4(quantized, global_scale, block_scales, output_type=torch.float32) # nvfp4
    elif quantized.dtype == torch.float8_e4m3fn:
        assert global_scale is not None, "fp8 requires global_scale"
        dequantized = ck.dequantize_per_tensor_fp8(quantized, global_scale, output_type=torch.float32) # fp8
    elif quantized.dtype == torch.int8:
        assert global_scale is not None, "int8 requires global_scale"
        dequantized = utils.dequantize_per_tensor_int8(quantized, global_scale) # fp8
    else:
        dequantized = quantized.to(dtype=torch.float32)

    amax = torch.amax(original.abs()).to(dtype=torch.float32)
    mse = torch.mean((original - dequantized).pow(2))
    psnr = 10 * torch.log10(amax.pow(2) / (mse + 1e-10)) if mse > 0 else float('inf')

    signal_power = torch.mean(original.pow(2))
    sqnr = 10 * torch.log10(signal_power / (mse + 1e-10)) if mse > 0 else float('inf')

    max_err = torch.max(torch.abs(original - dequantized))
    rel_max_err = (max_err / (original.abs().amax() + 1e-8))

    orig_flat = original.flatten()
    dequant_flat = dequantized.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_flat.unsqueeze(0), dequant_flat.unsqueeze(0)
    )

    return mse.item(), sqnr.item(), psnr.item(), cos_sim.item(), max_err.item(), rel_max_err.item(), amax.item()

def print_layer_header():
    print(f"{'layer_name':-^35} {'dtype':-^10} {'scale':-^10} {'mse':-^10} {'psnr':-^7} {'sqnr':-^7} {'cos_sim':-^8} {'relmaxerr':-^8}")

def print_layer_metrics(layer_name, original, quantized, global_scale=None, block_scales=None):
    mse, sqnr, psnr, cos_sim, max_err, rel_max_err, amax = get_metrics(original, quantized, global_scale, block_scales)
    gs = f"{fixed_e(global_scale, 4, 3):>10}" if global_scale !=None else f"{'':>10}"
    print(f"{layer_name:<35} {str(quantized.dtype).partition('.')[2][:10]:>10} {gs} {fixed_e(mse, 6, 3):>10} {psnr:>6.4f} {sqnr:>6.4f} {cos_sim*100:8.4f} {rel_max_err*100:>8.4f}")
