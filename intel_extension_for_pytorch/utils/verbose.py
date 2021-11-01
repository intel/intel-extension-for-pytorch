import torch
import intel_extension_for_pytorch._C as core

VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2
class verbose(object):
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        if self.level == VERBOSE_OFF:
            return
        try:
            st = torch._C._verbose.mkldnn_set_verbose(self.level)
            assert bool(st), "Failed to set Verbose mode of MKLDNN in PyTorch. Please consider to disable this verbose scope."
        except:
            pass
        st = core.mkldnn_set_verbose(self.level)
        assert bool(st), "Failed to set Verbose mode of MKLDNN in IPEX. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        core.mkldnn_set_verbose(VERBOSE_OFF)
        try:
            torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        except:
            pass
        return False

try:
    verbose_torch = torch.backends.mkldnn.verbose
    torch.backends.mkldnn.verbose = verbose
except:
    pass

