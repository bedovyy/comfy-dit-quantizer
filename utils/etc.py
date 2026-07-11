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

def get_metrics(original, dequantized):
    orig = original.to(torch.float32)
    dequant = dequantized.to(torch.float32)
    diff = orig - dequant

    mse = torch.mean(diff.pow(2))
    rmse = torch.sqrt(mse)

    signal_power = torch.mean(orig.pow(2))
    sqnr = 10 * torch.log10(signal_power / (mse + 1e-10)) if mse > 0 else torch.tensor(torch.inf)

    cos_sim = torch.nn.functional.cosine_similarity(orig.flatten(), dequant.flatten(), dim=0)

    amax = orig.abs().max()
    rel_max_err = (diff.abs().max() / (amax + 1e-8))

    #psnr = 10 * torch.log10(amax.pow(2) / (mse + 1e-10)) if mse > 0 else torch.tensor(torch.inf)

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "sqnr": sqnr.item(),
        "cos_sim": cos_sim.item(),
        "rel_max_err": rel_max_err.item(),
        "amax": amax.item()
    }

def print_layer_header():
    print(f"{'layer_name':-^60} {'qtype':-^12} {'mse':-^10} {'rmse':-^7} {'sqnr':-^7} {'cos_sim':-^8} {'relmaxerr':-^8}")

def print_layer_metrics(layer_name, qtype, original, dequantized):
    m = get_metrics(original, dequantized)
    mse, rmse, sqnr, cos_sim, rel_max_err, amax = m["mse"], m["rmse"], m["sqnr"], m["cos_sim"], m["rel_max_err"], m["amax"]
    if len(layer_name) > 60:
        layer_name = "..." + layer_name[-57:]
    print(f"{layer_name:<60} {str(qtype).split('.')[-1][:12]:>10} {fixed_e(mse, 6, 3):>10} {rmse:.5f} {sqnr:>6.4f} {cos_sim*100:8.4f} {rel_max_err*100:>8.4f}")
