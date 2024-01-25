from ..utils.utils import has_xpu
from ._lars import Lars

if has_xpu():
    from .xpu.ResourceApplyMomentum import FusedResourceApplyMomentum
