import torch

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
