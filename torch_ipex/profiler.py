import torch.autograd


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
        torch.autograd._enable_profiler(
            torch.autograd.ProfilerConfig(
                torch.autograd.ProfilerState.ITT,
                self.record_shapes,
                False,
                False)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.autograd._disable_profiler()
        return False
