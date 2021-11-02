from .. import _C
from enum import Enum

class XPU_BACKEND(Enum):
    GPU = 0
    CPU = 1
    AUTO = 2

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class VERBOSE_LEVEL(Enum):
    OFF = 0
    ON = 1
    ON_DETAIL = 2

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

def has_onedpl():
    return _C._is_onedpl_enabled()

def has_onemkl():
    return _C._is_onemkl_enabled()

def has_channels_last_1d():
    return _C._is_channels_last_1d_enabled()

def has_double_dtype():
    return not _C._is_double_disabled()

def get_warning_level():
    return _C._get_warning_level()

def get_xpu_backend():
    return XPU_BACKEND(_C._get_xpu_backend())

def set_xpu_backend(backend):
    if isinstance(backend, XPU_BACKEND):
        return _C._set_xpu_backend(backend.value)
    if isinstance(backend, str) and backend.isdecimal():
        value = int(backend)
        if XPU_BACKEND.has_value(value):
            return _C._set_xpu_backend(value)
    if isinstance(backend, int) and XPU_BACKEND.has_value(backend):
        return _C._set_xpu_backend(backend)
    raise RuntimeError("Unexpected XPU_BACKEND value {}!".format(backend))

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

class onednn_verbose(object):
    def __init__(self, level):
        self.level = VERBOSE_LEVEL.OFF
        if isinstance(level, VERBOSE_LEVEL):
            self.level = level
        elif isinstance(level, str) and level.isdecimal():
            value = int(level)
            if VERBOSE_LEVEL.has_value(value):
                self.level = VERBOSE_LEVEL(value)
            else:
                raise RuntimeError("Unexpected VERBOSE_LEVEL value {}!".format(level))
        elif isinstance(level, int) and VERBOSE_LEVEL.has_value(level):
            self.level = VERBOSE_LEVEL(level)
        else:
            raise RuntimeError("Unexpected VERBOSE_LEVEL value {}!".format(level))

    def __enter__(self):
        if self.level == VERBOSE_LEVEL.OFF:
            return
        st = _C._set_onednn_verbose(self.level.value)
        assert bool(st), "WARNING: Failed to turn on oneDNN verbose!"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _C._set_onednn_verbose(VERBOSE_LEVEL.OFF.value)
        return False
