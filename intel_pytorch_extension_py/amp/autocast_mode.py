import torch
import functools
import warnings
import numpy as np
#from torch._six import container_abcs, string_classes
import _torch_ipex as core
from .. import conf

class autocast(object):
    def __init__(self, enabled=True, configure=conf.AmpConf(torch.bfloat16)): 
        supported_dtype = [torch.float32, torch.bfloat16, torch.int8]
        if configure.dtype not in supported_dtype :
            warnings.warn("In CPU autocast, but the target dtype is not supported. Disable the autocast.")
            warnings.warn("Supported dtype input is: torch.float32, torch.bfloat16, torch.int8.")
            enabled = False
            configure = conf.AmpConf(torch.float32)
        self._enabled = enabled
        self._dtype = configure.dtype

    def __enter__(self):
        self.prev = core.is_autocast_enabled()
        self.prev_dtype = core.get_autocast_dtype()
        self.pre_calibration_state = core.get_int8_calibration()
        core.set_autocast_enabled(self._enabled)
        core.set_autocast_dtype(self._dtype)
        core.autocast_increment_nesting()
        if torch.int8 == self._dtype:
            core.disable_int8_calibration()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if core.autocast_decrement_nesting() == 0:
            core.clear_autocast_cache()
        core.set_autocast_enabled(self.prev)
        core.set_autocast_dtype(self.prev_dtype)
        if torch.int8 == self._dtype:
            if self.pre_calibration_state:
                core.enable_int8_calibration()
            else:
                core.disable_int8_calibration()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast

class calibrate(object):
    def __init__(self):
        self.pre_calibration_state = core.get_int8_calibration()

    def __enter__(self):
        self.prev = core.is_autocast_enabled()
        self.prev_dtype = core.get_autocast_dtype()
        core.set_autocast_enabled(True)
        core.set_autocast_dtype(torch.int8)
        core.autocast_increment_nesting()
        core.enable_int8_calibration()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if core.autocast_decrement_nesting() == 0:
            core.clear_autocast_cache()
        core.set_autocast_enabled(self.prev)
        core.set_autocast_dtype(self.prev_dtype)
        core.calibration_reset()
        if self.pre_calibration_state:
            core.enable_int8_calibration()
        else:
            core.disable_int8_calibration()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast