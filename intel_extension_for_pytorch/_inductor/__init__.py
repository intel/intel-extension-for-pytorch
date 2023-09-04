# from torch._inductor.codegen.common import register_backend_for_device

from . import xpu

from .xpu.codegen.triton import XPUTritonScheduling
from .xpu.codegen.wrapper import XPUTritonWrapperCodeGen

# Register triton XPU backend in PyTorch _inductor.
# register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
