from .autocast_mode import autocast
import intel_extension_for_pytorch


def get_autocast_xpu_dtype():
    return intel_extension_for_pytorch._C.get_autocast_xpu_dtype()


def is_autocast_xpu_enabled():
    return intel_extension_for_pytorch._C.is_autocast_xpu_enabled()


def set_autocast_xpu_enabled(enabled):
    return intel_extension_for_pytorch._C.set_autocast_xpu_enabled(enabled)


def set_autocast_xpu_dtype(dtype):
    return intel_extension_for_pytorch._C.set_autocast_xpu_dtype(dtype)
