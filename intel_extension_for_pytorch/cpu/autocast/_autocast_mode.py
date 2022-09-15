import torch
import intel_extension_for_pytorch._C as core

# Expand torch cpu autocast to support more data types in the future
class _autocast(torch.cpu.amp.autocast):
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
