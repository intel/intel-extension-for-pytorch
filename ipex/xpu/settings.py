from .. import _C
from enum import Enum

class XPU_BACKEND(Enum):
    GPU = 0
    CPU = 1
    AUTO = 2
    MAX = AUTO + 1

def has_onedpl():
    return _C._is_onedpl_enabled()

def has_onemkl():
    return _C._is_onemkl_enabled()

def has_double_dtype():
    return not _C._is_double_disabled()

def get_warning_level():
    return _C._get_warning_level()

def get_xpu_backend():
    return XPU_BACKEND(_C._get_xpu_backend())

def set_xpu_backend(backend : XPU_BACKEND):
    return _C._set_xpu_backend(backend.value)

def using_force_sync_exec():
    return _C._is_force_sync_exec()

def using_event_profiling():
    return _C._is_event_profiling_enabled()

def using_tile_partition():
    return _C._is_tile_partition_enabled()

def using_onednn_layout():
    return _C._is_onednn_layout_enabled()

def using_tf32_mode():
    return _C._is_tf32_mode_enabled()
