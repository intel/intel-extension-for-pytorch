from torch.backends import ContextProp, PropModule
from .. import _C


def enable_onednn_deterministic():
    _C._enable_onednn_deterministic()


def disable_onednn_deterministic():
    _C._disable_onednn_deterministic()
