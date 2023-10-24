from intel_extension_for_pytorch.utils.utils import has_cpu

if has_cpu():
    from .modules import FrozenBatchNorm2d
    from . import functional
