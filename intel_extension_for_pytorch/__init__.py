import re

import torch
try:
    import torchvision
except ImportError:
    pass  # skip if torchvision is not available

from ._version import __version__
from .utils import _custom_fx_tracer
from .cpu import _cpu_isa
_cpu_isa.check_minimal_isa_support()

torch_version = ''
ipex_version = ''
matches = re.match(r'(\d+\.\d+).*', torch.__version__)
if matches and len(matches.groups()) == 1:
  torch_version = matches.group(1)
matches = re.match(r'(\d+\.\d+).*', __version__)
if matches and len(matches.groups()) == 1:
  ipex_version = matches.group(1)
if torch_version == '' or ipex_version == '' or torch_version != ipex_version:
  print('ERROR! Intel® Extension for PyTorch* needs to work with PyTorch {0}.*, but PyTorch {1} is found. Please switch to the matching version and run again.'.format(ipex_version, torch.__version__))
  exit(127)


from . import cpu
from . import quantization
from . import nn
from . import jit

from . import tpp


from .utils.verbose import verbose
from .frontend import optimize, compile, enable_auto_channels_last, disable_auto_channels_last, enable_onednn_fusion, set_fp32_math_mode, get_fp32_math_mode, FP32MathMode, fast_bert
from .cpu._auto_kernel_selection import _enable_dnnl, _disable_dnnl, _using_dnnl

# for xpu
import intel_extension_for_pytorch.xpu
from . import optim
