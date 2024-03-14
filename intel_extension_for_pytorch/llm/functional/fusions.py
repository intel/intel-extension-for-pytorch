import torch
import sys
from intel_extension_for_pytorch.transformers.models.xpu.fusions.activation_fusion import (  # noqa F401
    silu_mul_xpu,
    gelu_mul_xpu,
    add_rms_norm_xpu,
    add_layer_norm_xpu,
)


def _get_function_from_device(device_type: str, f):
    assert device_type in [
        "cpu",
        "xpu",
    ], "The device is not in the supported device list."
    target_f_name = f.__name__ + "_" + device_type
    assert hasattr(
        sys.modules[__name__], target_f_name
    ), f"Target function {f.__name__} on {device_type} haven't implemented yet."
    target_f = getattr(sys.modules[__name__], target_f_name)
    return target_f


def silu_mul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None):
    f = _get_function_from_device(x.device.type, silu_mul)
    return f(x, y, out)


def gelu_mul(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor = None, approximate="none"
):
    f = _get_function_from_device(x.device.type, gelu_mul)
    return f(x, y, out, approximate)


def add_rms_norm(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    f = _get_function_from_device(x.device.type, add_rms_norm)
    return f(add, x, weight, bias, eps, add_back)


def add_layer_norm(
    add: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    add_back: bool,
):
    f = _get_function_from_device(x.device.type, add_layer_norm)
    return f(add, x, weight, bias, eps, add_back)
