import os
import torch
from .version import __version__
from .tensor import *
from .optim import *
from .ops import *
import _torch_ipex as core

DEVICE = 'dpcpp'

def enable_auto_optimization(mixed_dtype = None):
    r""" Enable auto-mixed-precision to improve performance.

    The auto-mixed-precision auto reorders the tensor to the specified low precision data type.
    You don't need to convert the input tensors and the model to the specified data type manually,
    the extension will do it automatically and then dispatch the extension backend to accelerate
    computation

    Args:
        mixed_dtype(torch.dtype): Auto reorder the input tensors to the specified low precision data type
            and dispatch to oneDNN backend for computation

    """
    if mixed_dtype != None:
        core.enable_auto_dnnl(True)
    enable_auto_mix_precision(mixed_dtype)

def get_auto_optimization():
    return get_auto_mix_precision

def enable_auto_mix_precision(mixed_dtype = torch.bfloat16):
    if mixed_dtype == torch.bfloat16:
        core.enable_mix_bf16_fp32()
    else:
        core.disable_mix_bf16_fp32()

def get_auto_mix_precision():
    if core.get_mix_bf16_fp32():
        return torch.bfloat16
    else:
        return None
