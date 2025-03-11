Intel® Extension for PyTorch\* - DeepSpeed\* Kernels
=====================================================
(intel_extension_for_pytorch.deepspeed module)

## Introduction
[DeepSpeed](https://github.com/microsoft/DeepSpeed)\* creates custom kernels for its feature support and performance optimizations. The DeepSpeed custom kernels for Intel XPU device are integrated into Intel® Extension for PyTorch\* under the ecological library category. It worths noting that the kernels are designed specifically for DeepSpeed\* therefore it is NOT necessarily common or validated when being used in scenarios other than DeepSpeed\*.

The DeepSpeed\* kernels module provides below custom kernels for DeepSpeed\*:
- quantization: including quantize/dequantize with fp32/fp16, etc
- transformer inference: including the bias GeGLU, layernorm, layernorm + residual, layernorm + store pre layernorm residual, RMS norm, pre RMS norm, vector add, MLP with fp16, MoE residual matmul, reset cache, release/retake workspace etc.

## Supported Platform
This module supports xpu device on Intel® Data Center GPU Max Series only.


