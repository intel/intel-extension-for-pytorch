# This Python file uses the following encoding: utf-8
import re

import torch
import warnings

try:
    import torchvision
except ImportError:
    pass  # skip if torchvision is not available

import os
import sys
import glob
import ctypes
import platform

from . import cpu
from . import xpu
from . import quantization
from . import nn
from . import jit
from . import optim
from . import fx
from . import _meta_registrations

try:
    from .cpu import tpp
except BaseException:
    warnings.warn(
        "Please install transformers repo when you want to use fast_bert API."
    )

from .frontend import optimize
from .cpu.transformers import _optimize_transformers
from .frontend import enable_auto_channels_last, disable_auto_channels_last
from .frontend import set_fp32_math_mode, get_fp32_math_mode, FP32MathMode
from .cpu._auto_kernel_selection import _enable_dnnl, _disable_dnnl, _using_dnnl
from .cpu.utils.verbose import verbose
from .cpu.tpp.fused_bert import fast_bert
from ._inductor.compiler import _set_compiler_backend, _get_compiler_backend, compile
from ._inductor.dynamo_backends import *
from .cpu.onednn_fusion import enable_onednn_fusion
from ._init_on_device import _IPEXOnDevice

from . import _C
from ._version import (
    __version__,
    __ipex_gitrev__,
    __torch_gitrev__,
    __gpu_onednn_gitrev__,
    __cpu_ideep_gitrev__,
    __build_type__,
)


# Path to folder containing CMake definitions for torch ipex package
cmake_prefix_path = os.path.join(os.path.dirname(__file__), "share", "cmake")


torch_version = ""
ipex_version = ""
matches = re.match(r"(\d+\.\d+).*", torch.__version__)
if matches and len(matches.groups()) == 1:
    torch_version = matches.group(1)
matches = re.match(r"(\d+\.\d+).*", __version__)
if matches and len(matches.groups()) == 1:
    ipex_version = matches.group(1)
if torch_version == "" or ipex_version == "" or torch_version != ipex_version:
    print(
        "ERROR! Intel® Extension for PyTorch* needs to work with PyTorch \
      {0}.*, but PyTorch {1} is found. Please switch to the matching version \
          and run again.".format(
            ipex_version, torch.__version__
        )
    )
    exit(127)


from .cpu.utils import _cpu_isa, _custom_fx_tracer

_cpu_isa.check_minimal_isa_support()


def version():
    print("intel_extension_for_pytorch version:          {}".format(__version__))
    print("intel_extension_for_pytorch git sha:          {}".format(__ipex_gitrev__))
    if len(__torch_gitrev__) != 0:
        print("torch version and sha:     {}".format(__torch_gitrev__))
    print("submodule oneDNN sha:      {}".format(__gpu_onednn_gitrev__))
    print("submodule ideep sha:      {}".format(__cpu_ideep_gitrev__))
