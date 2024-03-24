import torch
from ...utils.utils import has_xpu
from .overrides import override_size_asserts
from torch._inductor.codegen.common import register_backend_for_device
from .codegen.triton import XPUTritonScheduling
from .codegen.wrapper import XPUTritonWrapperCodeGen

from .lowering import *
from .fx_passes.fusion import *
from ._meta_registrations import *

if torch.xpu._is_compiled() and has_xpu():
    override_size_asserts()
    register_backend_for_device("xpu", XPUTritonScheduling, XPUTritonWrapperCodeGen)
