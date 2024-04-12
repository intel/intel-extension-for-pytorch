"""Utility functions for IPEX FP8 modules"""

import torch


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype"""
    return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


def cast_to_fp8(
    inp,
    fp8_meta_tensor,
    fp8_tensor,
    otype,
) -> torch.Tensor:
    """Cast input to FP8"""
    return torch.ops.torch_ipex.cast_to_fp8(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history[0],
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
    )


def cast_from_fp8(
    inp,
    fp8_meta_tensor,
    fp8_tensor,
    itype,
    otype,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.torch_ipex.cast_from_fp8(
        inp,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        itype,
        otype,
    )
