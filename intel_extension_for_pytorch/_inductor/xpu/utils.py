import functools

import torch


@functools.lru_cache(None)
def has_triton():
    if not torch.xpu.is_available():
        return False
    try:
        import triton

        return triton is not None
    except ImportError:
        return False
