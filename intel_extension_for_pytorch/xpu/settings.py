from .. import _C
from enum import Enum

def has_onemkl():
    return _C._is_onemkl_enabled()

def has_itt():
    return _C._itt_is_enabled()

def has_channels_last_1d():
    return _C._is_channels_last_1d_enabled()

def has_double_dtype():
    return not _C._is_double_disabled()

class EnumBase(Enum):
    @classmethod
    def convert(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str) and value.isdecimal():
            value = int(value)
        if isinstance(value, int) and cls.has_value(value):
            return cls(value)
        raise RuntimeError("Unexpected {} value {}!".format(cls, value))

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def get_value(cls, get_func):
        return cls(get_func())

    @classmethod
    def set_value(cls, set_func, value):
        set_func(cls.convert(value).value)

# Verbose Level
class VERBOSE_LEVEL(EnumBase):
    OFF = 0
    ON = 1

def get_verbose_level():
    return VERBOSE_LEVEL.get_value(_C._get_verbose_level)

def set_verbose_level(level):
    VERBOSE_LEVEL.set_value(_C._set_verbose_level, level)

# oneDNN Verbose
class ONEDNN_VERB_LEVEL(EnumBase):
    OFF = 0
    ON = 1
    ON_DETAIL = 2

def set_onednn_verbose(level):
    st = _C._set_onednn_verbose(ONEDNN_VERB_LEVEL.convert(level).value)
    assert bool(st), "WARNING: Failed to turn on oneDNN verbose!"

class onednn_verbose(object):
    def __init__(self, level):
        self.level = ONEDNN_VERB_LEVEL.convert(level)

    def __enter__(self):
        if self.level == ONEDNN_VERB_LEVEL.OFF:
            return self
        set_onednn_verbose(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_onednn_verbose(ONEDNN_VERB_LEVEL.OFF)
        return False

# oneMKL Verbose
class ONEMKL_VERB_LEVEL(EnumBase):
    OFF = 0
    ON = 1
    ON_SYNC = 2

def set_onemkl_verbose(level):
    st = _C._set_onemkl_verbose(ONEMKL_VERB_LEVEL.convert(level).value)
    assert bool(st), "WARNING: Failed to turn on oneMKL verbose!"

class onemkl_verbose(object):
    def __init__(self, level):
        self.level = ONEMKL_VERB_LEVEL.convert(level)

    def __enter__(self):
        if self.level == ONEMKL_VERB_LEVEL.OFF:
            return self
        set_onemkl_verbose(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_onemkl_verbose(ONEMKL_VERB_LEVEL.OFF)
        return False


# Sync Execution Mode
def using_xpu_sync_mode():
    return _C._is_xpu_sync_mode()

def enable_xpu_sync_mode():
    _C._enable_xpu_sync_mode()

def disable_xpu_sync_mode():
    _C._disable_xpu_sync_mode()

class xpu_sync_mode():
    def __init__(self):
        self.sync_exec = using_xpu_sync_mode()

    def __enter__(self):
        if not self.sync_exec:
            enable_xpu_sync_mode()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.sync_exec:
            disable_xpu_sync_mode()
        return False


# oneDNN Layout
def using_layout_opt():
    return _C._is_layout_opt_enabled()

def enable_layout_opt():
    _C._enable_layout_opt()

def disable_layout_opt():
    _C._disable_layout_opt()

class layout_opt():
    def __init__(self):
        self.layout_enabled = using_layout_opt()

    def __enter__(self):
        if not self.layout_enabled:
            enable_layout_opt()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.layout_enabled:
            disable_layout_opt()
        return False

# Tile Partition As Device
def using_tile_as_device():
    return _C._is_tile_as_device_enabled()


# # XPU Backend
# # NOTE: XPU Backend is not available yet.
# class XPU_BACKEND(EnumBase):
#     GPU = 0
#     CPU = 1
#     AUTO = 2
# 
# def get_xpu_backend():
#     return XPU_BACKEND.get_value(_C._get_xpu_backend)
# 
# def set_xpu_backend(backend):
#     XPU_BACKEND.set_value(_C._set_xpu_backend, backend)
# 
# # TF32 Execution Mode
# # NOTE: TF32 mode is not available yet.
# def using_tf32_mode():
#     return _C._is_tf32_mode_enabled()
# 
# def enable_tf32_mode():
#     _C._enable_tf32_mode()
# 
# def disable_tf32_mode():
#     _C._disable_tf32_mode()
# 
# class tf32_mode():
#     def __init__(self):
#         self.tf32_enabled = using_tf32_mode()
# 
#     def __enter__(self):
#         if not self.tf32_enabled:
#             enable_tf32_mode()
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if not self.tf32_enabled:
#             disable_tf32_mode()
#         return False
