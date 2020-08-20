import os
import torch
from .version import __version__
from .tensor import *
from .optim import *
from .ops import *
import _torch_ipex as core

DEVICE = 'dpcpp'

def enable_auto_optimization(mixed_dtype = None, train = False):
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
        core.enable_auto_dnnl()
    enable_auto_mix_precision(mixed_dtype, train)

def get_auto_optimization():
    return get_auto_mix_precision

def get_train():
    return core.get_train()

def enable_auto_mix_precision(mixed_dtype = torch.bfloat16, train = False):
    if mixed_dtype == torch.bfloat16:
        core.enable_mix_bf16_fp32()
        core.disable_mix_int8_fp32()
    elif mixed_dtype == torch.int8 or mixed_dtype == torch.uint8:
        core.enable_mix_int8_fp32()
        core.disable_mix_bf16_fp32()
    else:
        core.disable_mix_int8_fp32()
        core.disable_mix_bf16_fp32()
    core.set_execution_mode(train=train)

def get_auto_mix_precision():
    if core.get_mix_bf16_fp32():
        return torch.bfloat16
    elif core.get_mix_int8_fp32():
        return torch.int8
    else:
        return None

def calibration_reset():
    if core.get_int8_calibration():
        core.calibration_reset()
    else:
        raise ValueError("please first run enable_calibration before calibration reset")

class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator, copy form pytorch FW"""

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_context

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)
            while True:
                try:
                    with self:
                        x = next(gen)
                    yield x
                except StopIteration:
                    break
        return generator_context

class int8_calibration(_DecoratorContextManager):
    def __init__(self, file_name):
        self.configure_file = file_name

    def __enter__(self):
        if not core.get_mix_int8_fp32():
            raise ValueError("please first run enable_auto_mix_precision(torch.int8) before int8 calibration")
        core.enable_int8_calibration()

    def __exit__(self, *args):
        core.disable_int8_calibration()
        core.add_indicators()
        core.save_indicators_file(self.configure_file)
        return False

class int8_validate(_DecoratorContextManager):
    def __init__(self, file_name):
        self.configure_file = file_name

    def __enter__(self):
        if not core.get_mix_int8_fp32():
            raise ValueError("please first run enable_auto_mix_precision(torch.int8) before int8 validation")
        core.disable_int8_calibration()
        core.load_indicators_file(self.configure_file)

    def __exit__(self, *args):
        return False

