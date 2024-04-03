import functools
import multiprocessing
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
from time import sleep
from types import ModuleType

from typing import Union, Set
from functools import partial

import torch

from torch._inductor import config
from torch._inductor.codecache import (
    _async_compile_initializer,
    _compile_start,
    _worker_compile,
    _load_kernel,
    TritonFuture,
    AsyncCompile,
    caching_device_properties
)
from torch._dynamo.device_interface import get_interface_for_device


_pool_set: Set[ProcessPoolExecutor] = set()
class XPUAsyncCompile(AsyncCompile):
    def __init__(self):
        super().__init__()

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> ProcessPoolExecutor:
        # ensure properties have been calculated before processes
        # are forked
        caching_device_properties()
        assert config.compile_threads > 1
        orig_ppid = os.getpid()

        ctx = multiprocessing.get_context(config.worker_start_method)
        pool = ProcessPoolExecutor(
            config.compile_threads,
            mp_context=ctx,
            initializer=partial(_async_compile_initializer, orig_ppid),
        )

        global _pool_set
        _pool_set.add(pool)

        # when this pool is created in a subprocess object, the normal exit handler
        # doesn't run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers...
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    def triton(
        self, kernel_name: str, source_code: str, device_str: str = "xpu"
    ) -> Union[TritonFuture, ModuleType]:
        _compile_start()

        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device_str)
            device = torch.device(device_str, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device
            )
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)


XPUAsyncCompile.warm_pool()
