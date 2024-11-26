"""Utility functions for IPEX FP8 modules"""

import torch
from intel_extension_for_pytorch.frontend import _copy_model_and_optimizer
from .fp8 import FP8GlobalStateManager


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype"""
    return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


def cast_to_fp8(
    inp,
    fp8_meta_tensor,
    fp8_tensor,
    otype,
    out=None,
) -> torch.Tensor:
    """Cast input to FP8"""
    if out is not None:
        torch.ops.torch_ipex.cast_to_fp8_out(
            inp,
            fp8_meta_tensor.scale,
            out,
            fp8_meta_tensor.amax_history,
            fp8_meta_tensor.scale_inv,
            fp8_tensor,
            otype,
        )
        return None
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


def convert(model, optimizer, device="xpu"):
    from .linear import FP8Linear

    torch_modules = {
        torch.nn.Linear: FP8Linear,
    }
    if model.__class__ not in torch_modules.keys():
        return model

    module_attr = model.__constants__
    args = []
    for attr in module_attr:
        args.append(getattr(model, attr))
    new_m = torch_modules[model.__class__](
        *args, bias=True if model.bias is not None else False, device=device
    )

    for k, v in model.__dict__.items():
        if k == "_parameters":
            for p in list(v):
                setattr(new_m, p, getattr(model, p))
        else:
            new_m.__dict__[k] = v
    return new_m


def convert_rec(m, optimizer, device="xpu"):
    new_m = convert(m, optimizer, device)
    for name, sub_m in m.named_children():
        setattr(new_m, name, convert_rec(sub_m, optimizer, device)[0])
    return new_m, optimizer


def prepare_fp8(model, optimizer=None, device="xpu"):
    """Convert modules to FP8 modules (e.g, convert nn.Linear to FP8Linear) in the model."""
    FP8GlobalStateManager.set_fp8_device_type(device)
    optimized_model, optimized_optimizer = _copy_model_and_optimizer(model, optimizer)
    new_m, new_optimizer = convert_rec(optimized_model, optimized_optimizer, device)
    if optimizer is None:
        return new_m
    return new_m, new_optimizer
