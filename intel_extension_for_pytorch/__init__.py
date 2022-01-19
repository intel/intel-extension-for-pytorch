import torch
try:
    import torchvision
except ImportError:
    pass  # skip if torchvision is not available

from .version import __version__, __avx_version__

from .utils import _cpuinfo
_cpuinfo._check_avx_isa(__avx_version__)

from . import cpu
from . import quantization
from . import nn
from . import jit

from .utils.verbose import verbose
from .frontend import optimize, enable_onednn_fusion
