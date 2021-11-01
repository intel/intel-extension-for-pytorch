import os
import json
import warnings
import torch
try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from .version import __version__, __ipex_avx_version__

from . import cpu
from . import quantization
from . import nn

from .utils import verbose
from .frontend import optimize

def check_avx_isa(binary_isa):
    import intel_extension_for_pytorch._C as core
    err_msg = "The extension binary is {} while the current machine does not support it."
    if binary_isa == "AVX2":
        if not core._does_support_avx2():
            sys.exit(err_msg.format(binary_isa))
    elif binary_isa == "AVX512":
        if not core._does_support_avx512():
            sys.exit(err_msg.format(binary_isa))
    else:
        sys.exit("The extension only supports AVX2 and AVX512 now. The binary is not a correct version.")

check_avx_isa(__ipex_avx_version__)
