import torch
import functools
import warnings
import numpy as np
#from torch._six import container_abcs, string_classes
import _torch_ipex as core

def get_target_layout():
    return core.get_autocast_layout()

def set_target_layout(layout):
    core.set_autocast_layout(layout)

class autocast(object):
    def __init__(self, enabled=True, dtype=torch.float32): 
        supported_dtype = [torch.float32, torch.bfloat16, torch.int8]
        if dtype not in supported_dtype :
            warnings.warn("In CPU autocast, but the target dtype is not supported. Disable the autocast.")
            warnings.warn("Supported dtype input is: torch.float32, torch.bfloat16, torch.int8.")
            enabled = False
            dtype = torch.float32
        self._enabled = enabled
        self._dtype = dtype

    def __enter__(self):
        self.prev = core.is_autocast_enabled()
        self.prev_dtype = core.get_autocast_dtype()
        core.set_autocast_enabled(self._enabled)
        core.set_autocast_dtype(self._dtype)
        core.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if core.autocast_decrement_nesting() == 0:
            core.clear_autocast_cache()
        core.set_autocast_enabled(self.prev)
        core.set_autocast_dtype(self.prev_dtype)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast

