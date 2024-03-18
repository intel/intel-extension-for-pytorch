import deepspeed_ops


def ds_quantize_fp32(vals, groups, bits):
    return deepspeed_ops.ds_quantize_fp32(vals, groups, bits)
