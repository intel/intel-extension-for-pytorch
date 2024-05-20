import sys
from intel_extension_for_pytorch.transformers.models.cpu.fusions.mha_fusion import (  # noqa F401
    silu_mul_cpu,
    gelu_mul_cpu,
    add_rms_norm_cpu,
    add_layer_norm_cpu,
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
