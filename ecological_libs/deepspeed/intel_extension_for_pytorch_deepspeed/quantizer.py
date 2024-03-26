import quantization


def ds_quantize_fp32(vals, groups, bits):
    return quantization.ds_quantize_fp32(vals, groups, bits)
