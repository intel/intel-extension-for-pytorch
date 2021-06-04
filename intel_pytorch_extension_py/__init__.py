import os
import json
import warnings
import torch
from .version import __version__
from .tensor import *
from .optim import *
from .ops import *
import _torch_ipex as core
_C.enable_torch_ccl()

DEVICE = 'xpu:0'

class AmpConf(object):
    def __init__(self, mixed_dtype = torch.bfloat16, configure_file = None):
        self.dtype = mixed_dtype
        self.configure_file = configure_file

        if self.dtype != torch.bfloat16:
            _C.clear_indicators()
        # for int8 path, if user give a exited configure file, load it.
        if self.configure_file != None and self.dtype != torch.bfloat16:
            if os.path.exists(self.configure_file) and os.stat(self.configure_file).st_size != 0:
                with open(self.configure_file, 'r') as f:
                    configures = json.load(f)
                    _C.load_indicators_file(configures)
            else:
                assert False, 'Can not load a empty file or none existed file, plese first do calibartion step'

    # for int8 quantization, will save the date after doing calibration step.
    def save(self, configure_file):
        _C.add_indicators()
        configures = _C.get_int8_configures()
        with open(configure_file, 'w') as fp:
            json.dump(configures, fp, indent = 4)

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

def get_auto_mix_precision():
    if _C.get_mix_bf16_fp32():
        return torch.bfloat16
    elif _C.get_mix_int8_fp32():
        return torch.int8
    else:
        return None

def _enable_auto_optimization(mixed_dtype = None, train = False):
    if mixed_dtype != None:
        _C.enable_auto_dnnl()
    enable_auto_mixed_precision(mixed_dtype, train)

def enable_auto_mixed_precision(mixed_dtype = torch.bfloat16, train = False):
    r""" Enable auto-mixed-precision to improve performance for global scope.

    The auto-mixed-precision auto reorders the tensor to the specified low precision data type.
    You don't need to convert the input tensors and the model to the specified data type manually,
    the extension will do it automatically and then dispatch the extension backend to accelerate
    computation

    Args:
        mixed_dtype(torch.dtype): Auto reorder the input tensors to the specified low precision data type
            and dispatch to oneDNN backend for computation, can be torch.bfloat16 or None.
    """
    running_mode = 'training' if train else 'inference'
    AutoMixPrecision(AmpConf(mixed_dtype), running_mode).__enter__()

def _get_auto_optimization():
    return get_auto_mix_precision

def get_train():
    return _C.get_train()

class AutoMixPrecision(_DecoratorContextManager):
    def __init__(self, conf, running_mode = 'inference'):
        self.pre_mixed_dtype = get_auto_mix_precision()
        self.pre_running_mode = get_train()
        self.pre_calibration_state = _C.get_int8_calibration()
        self.mixed_dtype = conf.dtype
        self.running_mode = running_mode

    def __enter__(self):
        if self.mixed_dtype == torch.bfloat16:
            _C.enable_mix_bf16_fp32()
            _C.disable_mix_int8_fp32()
        elif self.mixed_dtype == torch.int8:
            _C.enable_mix_int8_fp32()
            _C.disable_mix_bf16_fp32()
            if self.running_mode == 'inference':
                _C.disable_int8_calibration()
            elif self.running_mode == 'calibration':
                _C.enable_int8_calibration()
            else:
                assert False, 'int8 quantization only suport inference and calibration running mode'
        else:
            _C.disable_mix_int8_fp32()
            _C.disable_mix_bf16_fp32()
        _C.set_execution_mode(train = True if self.running_mode == 'training' else False)

    def __exit__(self, *args):
        if self.mixed_dtype == torch.int8:
            if self.running_mode == 'calibration':
                _C.calibration_reset()
        # restore previous state
        if self.pre_calibration_state:
            _C.enable_int8_calibration()
        else:
            _C.disable_int8_calibration()
        if self.pre_mixed_dtype == torch.bfloat16:
            _C.enable_mix_bf16_fp32()
            _C.disable_mix_int8_fp32()
        elif self.pre_mixed_dtype == torch.int8:
            _C.enable_mix_int8_fp32()
            _C.disable_mix_bf16_fp32()
        else:
            _C.disable_mix_int8_fp32()
            _C.disable_mix_bf16_fp32()
        _C.set_execution_mode(train = self.pre_running_mode)
