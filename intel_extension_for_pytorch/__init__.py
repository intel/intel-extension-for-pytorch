import torch
try:
    import torchvision
except ImportError:
    pass  # skip if torchvision is not available

from .version import __version__, __avx_version__

torch_version = ''
ipex_version = ''
import re
matches = re.match('(\d+\.\d+).*', torch.__version__)
if matches and len(matches.groups()) == 1:
  torch_version = matches.group(1)
matches = re.match('(\d+\.\d+).*', __version__)
if matches and len(matches.groups()) == 1:
  ipex_version = matches.group(1)
if torch_version == '' or ipex_version == '' or torch_version != ipex_version:
  print('ERROR! Intel® Extension for PyTorch* needs to work with PyTorch {0}.*, but PyTorch {1} is found. Please switch to the matching version and run again.'.format(ipex_version, torch.__version__))
  exit(127)

from .utils import _cpuinfo
_cpuinfo._check_avx_isa(__avx_version__)

from . import cpu
from . import quantization
from . import nn
from . import jit

from .utils.verbose import verbose
from .frontend import optimize, enable_onednn_fusion
