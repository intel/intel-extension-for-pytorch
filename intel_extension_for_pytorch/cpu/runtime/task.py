import torch
import functools
import warnings
import numpy as np
import intel_extension_for_pytorch as ipex
from .cpupool import CPUPool

class Task(object):
    def __init__(self, module, cpu_pool: CPUPool):
        self.cpu_pool = cpu_pool
        assert type(self.cpu_pool) is CPUPool
        if isinstance(module, torch.jit.ScriptModule):
            self._task = ipex._C.TaskModule(module._c, self.cpu_pool.core_ids, True)
        else:
            self._task = ipex._C.TaskModule(module, self.cpu_pool.core_ids)

    def __call__(self, *args, **kwargs):
        # async execution
        return self._task.run_async(*args, **kwargs)

    def run_sync(self, *args, **kwargs):
        # sync execution
        return self._task.run_sync(*args, **kwargs)
