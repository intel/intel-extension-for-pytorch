import torch
from functools import wraps

# Wrap torch.jit.trace to disable autocast cache when using torch.jit.trace
# within the scope of torch.cpu.amp.autocast.
# See https://github.com/pytorch/pytorch/pull/63552 for more information.
def disable_autocast_cache(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        prev = torch.is_autocast_cache_enabled()
        # Disable autocast cache
        if torch.is_autocast_cpu_enabled():
            torch.set_autocast_cache_enabled(False)
        traced = f(*args, **kwargs)
        torch.set_autocast_cache_enabled(prev)
        return traced
    return wrapper

torch.jit.trace = disable_autocast_cache(torch.jit.trace)
