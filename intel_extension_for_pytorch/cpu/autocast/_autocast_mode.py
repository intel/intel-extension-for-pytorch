import torch
import intel_extension_for_pytorch._C as core
import warnings
from typing import Any, Optional
from torch.types import _dtype

# Expand torch.amp.autocast_mode.autocast to support both torch.bfloat16 and torch.float16 on cpu.
class _mode_autocast(torch.amp.autocast_mode.autocast):
    def __init__(self, device_type : str,
                 dtype : Optional[_dtype] = None,
                 enabled : bool = True,
                 cache_enabled : Optional[bool] = None):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            self.fast_dtype = dtype
            # TODO: support get_autocast_gpu/cpu_dtype
            assert dtype is not None
            return
        self.device = device_type
        if self.device == 'cuda':
            self.fast_dtype = torch.get_autocast_gpu_dtype()
        elif self.device == 'cpu':
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        elif self.device == 'xpu':
            self.fast_dtype = torch.xpu.get_autocast_xpu_dtype()  # type: ignore[attr-defined]
        else:
            raise RuntimeError('User specified autocast device_type must be \'cuda\' or \'cpu\'')
        self._cache_enabled = torch.is_autocast_cache_enabled()
        if torch.cuda.amp.common.amp_definitely_not_available() and self.device == 'cuda':
            warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
            enabled = False
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled

        if self.device == 'cpu':
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = 'In CPU autocast, but the target dtype is not supported. Disabling autocast.\n'
                error_message += 'CPU Autocast only supports dtype of torch.bfloat16 and torch.float16 currently.'
                warnings.warn(error_message)
                enabled = False
        if self.device == 'xpu':
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = 'In XPU autocast, but the target dtype is not supported. Disabling autocast.\n'
                error_message += 'XPU Autocast only supports dtype of torch.bfloat16 currently.'
                warnings.warn(error_message)
                enabled = False
        if self.device == 'cuda':
            if self.fast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError('Current CUDA Device does not support bfloat16. Please switch dtype to float16.')
        self._enabled = enabled

# same as torch.cpu.amp.autocast 
class autocast_cpu(_mode_autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("cpu", args...)``
    """
    def __init__(self, enabled : bool = True, dtype : torch.dtype = torch.bfloat16, cache_enabled : bool = True):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cpu"
            self.fast_dtype = dtype
            return
        super().__init__("cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Expand torch.cpu.amp.autocast to support both torch.bfloat16 & torch.half 
# and support the disabling of cache_enabled for autocast within jit.trace.
class _autocast(autocast_cpu):
    def __enter__(self):
        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        self.prev = torch.is_autocast_cpu_enabled()
        self.prev_fast_dtype = core.get_autocast_dtype()
        torch.set_autocast_cpu_enabled(self._enabled)
        core.set_autocast_dtype(self.fast_dtype)
        torch.autocast_increment_nesting()
        torch.set_autocast_cache_enabled(self._cache_enabled)

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            core.clear_autocast_cache()
            torch.clear_autocast_cache()
        torch.set_autocast_cpu_enabled(self.prev)
        core.set_autocast_dtype(self.prev_fast_dtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)
        return False


torch.cpu.amp.autocast = _autocast
