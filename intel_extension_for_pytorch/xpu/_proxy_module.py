import torch
import intel_extension_for_pytorch._C


# utils function to define base object proxy
def _proxy_module(name: str) -> type:
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate proxy base class {}.".format(class_name)
            + "\nIntel_extension_for_pytorch not compiled with XPU enabled."
        )

    return type(name, (object,), {"__init__": init_err})


def _register_proxy(module: str):
    if not hasattr(intel_extension_for_pytorch._C, module):
        intel_extension_for_pytorch._C.__dict__[module] = _proxy_module(module)


def _register_proxy_ops(module: str):
    if not hasattr(torch.ops.torch_ipex, module):
        torch.ops.torch_ipex.__dict__[module] = _proxy_module(module)


class proxy_math_mode(object):
    FP32 = -1
    TF32 = -2
    BF32 = -3


class proxy_compute_eng(object):
    RECOMMEND = -1
    BASIC = -2
    ONEDNN = -3
    ONEMKL = -4
    XETLA = -5


# --- [ CPU proxys:
_register_proxy_ops("interaction_forward")


if not hasattr(intel_extension_for_pytorch._C, "FP32MathMode"):
    intel_extension_for_pytorch._C.__dict__["FP32MathMode"] = proxy_math_mode


# --- [ XPU proxys:
_register_proxy("ShortStorageBase")
_register_proxy("CharStorageBase")
_register_proxy("IntStorageBase")
_register_proxy("LongStorageBase")
_register_proxy("BoolStorageBase")
_register_proxy("HalfStorageBase")
_register_proxy("DoubleStorageBase")
_register_proxy("FloatStorageBase")
_register_proxy("BFloat16StorageBase")
_register_proxy("QUInt8StorageBase")
_register_proxy("QInt8StorageBase")
_register_proxy("_XPUStreamBase")
_register_proxy("_XPUEventBase")


if not hasattr(intel_extension_for_pytorch._C, "XPUFP32MathMode"):
    intel_extension_for_pytorch._C.__dict__["XPUFP32MathMode"] = proxy_math_mode


if not hasattr(intel_extension_for_pytorch._C, "XPUComputeEng"):
    intel_extension_for_pytorch._C.__dict__["XPUComputeEng"] = proxy_compute_eng
