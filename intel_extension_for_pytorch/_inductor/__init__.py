from .overrides import override_decode_device
from .utils import has_triton

if has_triton():
    # Here we have to override decode_device since it is hardcode with CUDA in
    # PyTorch inductor.
    override_decode_device()


# We should register XPU Triton backend after override Inductor utils functions.
from torch._inductor.codegen.common import register_backend_for_device

from .codegen.triton import XPUTritonScheduling
from .codegen.wrapper import XPUTritonWrapperCodeGen

# Register triton XPU backend in PyTorch _inductor.
register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
