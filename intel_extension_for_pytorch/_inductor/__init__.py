from ..utils.utils import has_xpu

if has_xpu():
    from . import xpu
