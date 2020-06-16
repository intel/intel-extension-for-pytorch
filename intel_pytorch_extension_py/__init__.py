import os
import torch
from .version import __version__
from .tensor import *
from .optim import *
from .ops import *
import _torch_ipex as core


DEVICE = 'dpcpp'

def get_auto_optimization():
    return core.get_auto_dnnl()

def enable_auto_optimization(enable = True):
    if enable:
        core.enable_auto_dnnl()
    else:
        core.disable_auto_dnnl()

def get_auto_mix_precision(bf16 = True):
    return core.get_mix_bf16_fp32()

def enable_auto_mix_precision(bf16 = True):
    if bf16:
        core.enable_mix_bf16_fp32()
    else:
        core.disable_mix_bf16_fp32()
