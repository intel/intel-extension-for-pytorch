# coding: utf-8
from . import optim
from .frontend import optimize
import intel_extension_for_pytorch.xpu
from .cpu._auto_kernel_selection import _enable_dnnl, _disable_dnnl, _using_dnnl
from .frontend import enable_onednn_fusion, set_fp32_math_mode, get_fp32_math_mode, FP32MathMode
from .utils.verbose import verbose
import os
import torch
try:
    import torchvision
except ImportError:
    pass  # skip if torchvision is not available
from . import _C
# TODO: will uniform here after setup.py is uniformed
from ._version import (__version__, __ipex_gitrev__,
                    __torch_gitrev__, __gpu_onednn_gitrev__)
from .utils import _cpu_isa, _custom_fx_tracer
_cpu_isa.check_minimal_isa_support()

def version():
    print("intel_extension_for_pytorch version:          {}".format(__version__))
    print("intel_extension_for_pytorch git sha:          {}".format(__ipex_gitrev__))
    print("torch version and sha:     {}".format(__torch_gitrev__))
    print("submodule oneDNN sha:      {}".format(__gpu_onednn_gitrev__))


def _find_dpcpp_home():
    pass


def _here_paths():
    here = os.path.abspath(__file__)
    ipex_path = os.path.dirname(here)
    return ipex_path


def include_paths():
    '''
    Get the include paths required to build a C++ extension.

    Returns:
        A list of include path strings.
    '''
    ipex_path = _here_paths()
    lib_include = os.path.join(ipex_path, 'include')
    paths = [
        lib_include,
        # os.path.join(lib_include, 'more')
    ]
    return paths


def library_paths():
    '''
    Get the library paths required to build a C++ extension.

    Returns:
        A list of library path strings.
    '''
    ipex_path = _here_paths()
    lib_path = os.path.join(ipex_path, 'lib')
    paths = [
        lib_path,
    ]
    return paths


# Path to folder containing CMake definitions for torch ipex package
cmake_prefix_path = os.path.join(os.path.dirname(__file__), 'share', 'cmake')


# For CPU
torch_version = ''
ipex_version = ''
import re
matches = re.match('(\d+\.\d+).*', torch.__version__) # noqa W605
if matches and len(matches.groups()) == 1:
    torch_version = matches.group(1)
matches = re.match('(\d+\.\d+).*', __version__) # noqa W605
if matches and len(matches.groups()) == 1:
    ipex_version = matches.group(1)
if torch_version == '' or ipex_version == '' or torch_version != ipex_version:
    print('ERROR! Intel® Extension for PyTorch* needs to work with PyTorch \
      {0}.*, but PyTorch {1} is found. Please switch to the matching version \
          and run again.'.format(ipex_version, torch.__version__))
    exit(127)
