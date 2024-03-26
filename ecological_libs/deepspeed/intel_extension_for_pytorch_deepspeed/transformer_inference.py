import transformer_inference


def ds_bias_gelu_fp32(vals, bias):
    return transformer_inference.bias_gelu_fp32(vals, bias)
