import torch
import intel_extension_for_pytorch._C as core

class _profiler(torch.profiler.profile):
    def __enter__(self, *args):
        if(self.activities):
            for i in self.activities:
                if(i is torch.profiler.ProfilerActivity.CPU):
                    core.set_profile_op_enabled(True)
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        if(self.activities):
            for i in self.activities:
                if(i is torch.profiler.ProfilerActivity.CPU):
                    core.set_profile_op_enabled(False)

torch.profiler.profile = _profiler
