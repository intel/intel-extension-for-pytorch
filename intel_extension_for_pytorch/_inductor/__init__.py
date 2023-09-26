import torch
from torch._inductor.codegen.common import register_backend_for_device

from .codegen.triton import XPUTritonScheduling
from .codegen.wrapper import XPUTritonWrapperCodeGen

# Register triton XPU backend in PyTorch _inductor.
if torch.xpu.is_available():
    register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
