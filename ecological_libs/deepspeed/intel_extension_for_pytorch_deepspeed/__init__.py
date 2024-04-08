import sys
import torch
import intel_extension_for_pytorch

current_module = sys.modules[__name__]
intel_extension_for_pytorch._register_extension_module("deepspeed", current_module)

from .quantizer import quantize, dequantize, Symmetric, Asymmetric, ds_quantize_fp16, ds_quantize_fp32
from .transformer_inference import (
    gated_activation,
    bias_gelu_fp32,
    layer_norm,
    _layer_norm_residual,
    layer_norm_residual_store_pre_ln_res,
    rms_norm,
    pre_rms_norm,
    residual_add_bias_fp16,
    residual_add_bias_fp32,
    residual_add_bias_bf16,
    bias_gelu_fp16,
    bias_gelu_fp32,
    bias_add_fp16,
    bias_add_fp32,
    bias_add_bf16,
    bias_relu_fp16,
    bias_relu_fp32,
    bias_relu_bf16,
    bias_gelu_fp16,
    bias_gelu_fp32,
    bias_gelu_bf16,
)

