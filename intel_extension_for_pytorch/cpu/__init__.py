from ..utils.utils import has_cpu

from . import runtime
from . import autocast
from . import auto_ipex

if has_cpu():
    from . import comm
