import warnings
from .autocast_mode import autocast, custom_fwd, custom_bwd
import intel_extension_for_pytorch


def get_autocast_xpu_dtype():
    warnings.warn(
        "get_autocast_xpu_dtype is deprecated. Please use torch.get_autocast_dtype('xpu') instead."
    )
    return intel_extension_for_pytorch._C.get_autocast_xpu_dtype()


def is_autocast_xpu_enabled():
    warnings.warn(
        "is_autocast_xpu_enabled is deprecated. Please use torch.is_autocast_enabled('xpu') instead."
    )
    return intel_extension_for_pytorch._C.is_autocast_xpu_enabled()


def set_autocast_xpu_enabled(enabled):
    warnings.warn(
        "set_autocast_xpu_enabled is deprecated. Please use torch.set_autocast_enabled('xpu', enabled) instead."
    )
    return intel_extension_for_pytorch._C.set_autocast_xpu_enabled(enabled)


def set_autocast_xpu_dtype(dtype):
    warnings.warn(
        "set_autocast_xpu_dtype is deprecated. Please use torch.set_autocast_dtype('xpu', dtype) instead."
    )
    return intel_extension_for_pytorch._C.set_autocast_xpu_dtype(dtype)
