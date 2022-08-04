import torch.autograd
from contextlib import contextmanager

try:
    from .._C import _itt
except ImportError:
    class _ITTStub(object):
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError("intel_extension_for_pytorch is not build with ITT.")

        rangePush = _fail
        rangePop = _fail
        mark = _fail

    _itt = _ITTStub()

__all__ = ['range_push', 'range_pop', 'mark']


def range_push(msg):
    """
    Arguments:
      msg (string): ASCII message to associate with range
    """
    return _itt.rangePush(msg)


def range_pop():
    """
    """
    return _itt.rangePop()


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.
    Arguments:
      msg (string): ASCII message to associate with the event.
    """
    return _itt.mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an ITT range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().
    Args:
        msg (string): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    yield
    range_pop()


class emit_itt(object):
    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("ITT annotation context manager is not reentrant")
        self.entered = True
        torch.autograd._enable_profiler_legacy(
            torch.autograd.ProfilerConfig(
                torch.autograd.ProfilerState.ITT,
                self.record_shapes,
                False,
                False,
                False,
                False)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.autograd._disable_profiler_legacy()
        return False
