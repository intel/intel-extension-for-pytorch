import quantization

Symmetric = quantization.QuantizationType.Symmetric
Asymmetric = quantization.QuantizationType.Asymmetric

def quantize(input_vals, groups, numBits, quantType):
    return quantization.quantize(input_vals, groups, numBits, quantType)

def dequantize(activations, params, num_groups, q_bits, quant_type):
    return quantization.dequantize(activations, params, num_groups, q_bits, quant_type)

def ds_quantize_fp16(vals, groups, bits):
    return quantization.ds_quantize_fp16(vals, groups, bits)

def ds_quantize_fp32(vals, groups, bits):
    return quantization.ds_quantize_fp32(vals, groups, bits)

