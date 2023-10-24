import functools
import multiprocessing
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
from time import sleep

import torch

from torch._inductor import config
from torch._inductor.codecache import (
    _compile_start,
    _worker_compile,
    _load_kernel,
    TritonFuture,
    AsyncCompile,
)


class XPUAsyncCompile(AsyncCompile):
    def __init__(self):
        super().__init__()

    @staticmethod
    @functools.lru_cache(1)
    def process_pool():
        assert config.compile_threads > 1
        orig_ppid = os.getpid()

        # if this process dies abnormally (e.g. segfault)
        # it will not shut down the workers. Instead
        # the workers will have their parent reassigned to the
        # init process. This launches a separate thread to
        # watch for the worker getting reassigned,
        # and cleans it up in this case.
        def init():
            def run():
                while True:
                    sleep(1)
                    if orig_ppid != os.getppid():
                        os.kill(os.getpid(), signal.SIGKILL)

            global _watchdog_thread
            _watchdog_thread = Thread(target=run, daemon=True)
            _watchdog_thread.start()

        # we rely on 'fork' because we cannot control whether users
        # have an `if __name__ == '__main__'` in their main process.
        fork_context = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(
            config.compile_threads, mp_context=fork_context, initializer=init
        )
        # when this pool is created in a subprocess object, the normal exit handler
        # doesn't run, and we need to register our own handler.
        # exitpriority has to be high, because another one of the finalizers will
        # kill the worker thread that sends the shutdown message to the workers...
        multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)
        return pool

    def triton(self, kernel_name, source_code):
        _compile_start()

        if config.compile_threads > 1:
            device = torch.device("xpu", torch.xpu.current_device())
            cc = None
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device
            )
            return TritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)


XPUAsyncCompile.warm_pool()
