import torch
import intel_extension_for_pytorch._C as core
from enum import IntEnum

class LowPrecisionMode(IntEnum):
    BF32 = int(core.IPEXLowPrecisionMode.BF32)
    FP32 = int(core.IPEXLowPrecisionMode.FP32)

def set_fp32_low_precision_mode(mode=LowPrecisionMode.BF32):
    r"""
    Enable or disable implicit data type conversion.
    If mode is LowPrecisionMode.FP32 which means to disable the oneDNN fpmath mode.
    If mode is LowPrecisionMode.BF32 which means to enable the oneDNN fpmath mode by down convert to bfloat16 implicitly.

    Args:
        mode (LowPrecisionMode): Only works for ``LowPrecisionMode.FP32`` and ``LowPrecisionMode.BF32``.
            oneDNN fpmath mode will be disabled by default if dtype is set to ``LowPrecisionMode.FP32``.
            The implicit FP32 to BF16 data type conversion will be enabled if dtype is set to ``LowPrecisionMode.BF32`.

    Examples:

        >>> import intel_extension_for_pytorch as ipex
        >>> # to enable the implicit data type conversion
        >>> ipex.backends.cpu.set_fp32_low_precision_mode(mode=ipex.LowPrecisionMode.BF32)
        >>> # to disable the implicit data type conversion
        >>> ipex.backends.cpu.set_fp32_low_precision_mode(mode=ipex.LowPrecisionMode.FP32)
    """

    if mode == LowPrecisionMode.BF32:
        core.set_fp32_low_precision_mode(core.IPEXLowPrecisionMode.BF32)
    elif mode == LowPrecisionMode.FP32:
        core.set_fp32_low_precision_mode(core.IPEXLowPrecisionMode.FP32)
    else:
        warnings.warn("IPEX does not support mode except LowPrecisionMode.FP32 and LowPrecisionMode.BF32 for fpmath_mode.")


def get_fp32_low_precision_mode():
    r"""
    Get the current fpmath_mode setting.

    Returns:
        Fpmath mode
        The value will be ``LowPrecisionMode.FP32`` or ``LowPrecisionMode.BF32``.
        ``LowPrecisionMode.FP32`` means implicit down-conversion is disabled,
        while ``LowPrecisionMode.BF32`` means implicit down-conversions from f32 to bf16/f16 or compatible FP type is allowed.

    Examples:

        >>> import intel_extension_for_pytorch as ipex
        >>> # to get the current fpmath mode
        >>> ipex.backends.cpu.get_fp32_low_precision_mode()
    """

    return core.get_fp32_low_precision_mode()

