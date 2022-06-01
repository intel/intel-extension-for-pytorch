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
        return set_func(cls.convert(value).value)

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
    st = ONEDNN_VERB_LEVEL.set_value(_C._set_onednn_verbose, level)
    assert bool(st), "WARNING: Failed to turn on oneDNN verbose!"

class onednn_verbose(object):
    def __init__(self, level):
        self.level = ONEDNN_VERB_LEVEL.convert(level)

    def __enter__(self):
        if self.level != ONEDNN_VERB_LEVEL.OFF:
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
    st = ONEMKL_VERB_LEVEL.set_value(_C._set_onemkl_verbose, level)
    assert bool(st), "WARNING: Failed to turn on oneMKL verbose!"

class onemkl_verbose(object):
    def __init__(self, level):
        self.level = ONEMKL_VERB_LEVEL.convert(level)

    def __enter__(self):
        if self.level != ONEMKL_VERB_LEVEL.OFF:
            set_onemkl_verbose(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_onemkl_verbose(ONEMKL_VERB_LEVEL.OFF)
        return False


# FP32 math mode
class FP32_MATH_MODE(EnumBase):
    FP32 = 0
    TF32 = 1
    BF32 = 2

def get_fp32_math_mode():
    return FP32_MATH_MODE.get_value(_C._get_fp32_math_mode)

def set_fp32_math_mode(math_mode):
    st = FP32_MATH_MODE.set_value(_C._set_fp32_math_mode, math_mode)
    assert bool(st), "WARNING: Failed to set FP32 math mode!"

class fp32_math_mode(object):
    def __init__(self, math_mode):
        self.math_mode = FP32_MATH_MODE.convert(math_mode)

    def __enter__(self):
        current_math_mode = get_fp32_math_mode()
        if self.math_mode != current_math_mode:
            set_fp32_math_mode(self.math_mode)
            self.math_mode = current_math_mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_fp32_math_mode(self.math_mode)
        return False


# Basic ONOFF
class ONOFF():
    def __init__(self, checker, enable, disable):
        self._init_status = checker()
        self._enabled = True
        self._disabled = False
        self._enable_fn = enable
        self._disable_fn = disable

    def __enter__(self):
        if self._init_status == self._disabled:
            self._enable_fn()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._init_status == self._disabled:
            self._disable_fn()
        return False


# Sync Execution Mode
def using_sync_mode():
    return _C._is_sync_mode()

def enable_sync_mode():
    _C._enable_sync_mode()

def disable_sync_mode():
    _C._disable_sync_mode()

class sync_mode(ONOFF):
    def __init__(self):
        super().__init__(using_sync_mode, enable_sync_mode, disable_sync_mode)


# [EXPERIMENTAL] oneDNN Layout
def using_onednn_layout():
    return _C._is_onednn_layout_enabled()

def enable_onednn_layout():
    _C._enable_onednn_layout()

def disable_onednn_layout():
    _C._disable_onednn_layout()

class onednn_layout(ONOFF):
    def __init__(self):
        super().__init__(using_onednn_layout, enable_onednn_layout, disable_onednn_layout)


# Simple Trace
def using_simple_trace():
    return _C._is_simple_trace_enabled()

def enable_simple_trace():
    _C._enable_simple_trace()

def disable_simple_trace():
    _C._disable_simple_trace()

class simple_trace(ONOFF):
    def __init__(self):
        super().__init__(using_simple_trace, enable_simple_trace, disable_simple_trace)


# Tile Partition As Device
def using_tile_as_device():
    return _C._is_tile_as_device_enabled()

# XPU Backend
# NOTE: XPU Backend is not available yet.
class XPU_BACKEND(EnumBase):
    GPU = 0
    CPU = 1
    AUTO = 2

def get_backend():
    return XPU_BACKEND.get_value(_C._get_backend)

def set_backend(backend):
    st = XPU_BACKEND.set_value(_C._set_backend, backend)
    assert bool(st), "WARNING: Failed to set XPU backend!"
