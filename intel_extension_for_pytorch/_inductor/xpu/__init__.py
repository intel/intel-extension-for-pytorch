import torch
from .overrides import override_size_asserts
from torch._inductor.codegen.common import register_backend_for_device
from .codegen.triton import XPUTritonScheduling
from .codegen.wrapper import XPUTritonWrapperCodeGen

if torch.xpu.is_available():
    override_size_asserts()
    register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
