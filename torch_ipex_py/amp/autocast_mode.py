import torch
import intel_extension_for_pytorch._C as core


class _autocast_bf16(torch.cpu.amp.autocast):
    def __enter__(self):
        self.prev = torch.is_autocast_cpu_enabled()
        self.prev_dtype = torch.get_autocast_cpu_dtype()
        torch.set_autocast_cpu_enabled(self._enabled)
        core.set_autocast_dtype(self._dtype)
        torch.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            core.clear_autocast_cache()
        torch.set_autocast_cpu_enabled(self.prev)
        core.set_autocast_dtype(self.prev_dtype)
        return False


torch.cpu.amp.autocast = _autocast_bf16
