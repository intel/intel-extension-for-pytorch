r"""
This package is used to lazily initialize XPU, so split it from __init__.py to avoid circular import.
"""

from .. import _C
import traceback
import threading
from typing import List


_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # invoke these after initialization occurs
_is_in_bad_fork = getattr(_C, "_xpu_isInBadFork", lambda: False)


class _LazySeedTracker:
    # We only track the latest seed given by 'manual_seed_all' or 'manual_seed'.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> List:
        return self.call_order


_lazy_seed_tracker = _LazySeedTracker()


def is_initialized():
    r"""Returns whether XPU state has been initialized."""
    return _initialized and not _is_in_bad_fork()


class DeferredXPUCallError(Exception):
    pass


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # This test was was protected via GIL. Double-check whether XPU has
        # already been initialized. If a thread acquired the lock first,
        # it will do an initialization. When the other threads get the lock,
        # they will find XPU has been initialized.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL.
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize XPU in forked subprocess. To use XPU with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(_C, "_getDeviceCount"):
            raise AssertionError("IPEX not compiled with XPU enabled")
        # This function detects bad fork processing and throws if there's a device
        # initialization error, no XPUs are found or any other error occurs
        _C._initExtension()
        # Some of the queued calls in _queued_calls[] may reentrantly call
        # _lazy_init(). We must prevent multiple initializations. In that case
        # just return early without initializeing to avoid a deadlock.
        _tls.is_initializing = True

        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"XPU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"XPU call was originally invoked at:\n\n{orig_traceback}"
                    )
                    raise DeferredXPUCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))
