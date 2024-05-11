from intel_extension_for_pytorch.utils.utils import has_cpu

from . import functional

if has_cpu():
    from .modules import FrozenBatchNorm2d
