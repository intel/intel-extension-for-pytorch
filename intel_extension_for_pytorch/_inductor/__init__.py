from ..utils.utils import has_xpu, has_cpu

if has_xpu():
    from . import xpu
if has_cpu():
    from . import cpu
