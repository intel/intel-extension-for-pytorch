# coding: utf-8
from enum import Enum
from functools import lru_cache
from typing_extensions import deprecated
import torch
from .. import _C
from .. import frontend
import intel_extension_for_pytorch  # noqa F401
from .intrinsic import clip_grad_norm_  # noqa F401


def has_onemkl():
    return _C._is_onemkl_enabled()


@deprecated(
    "`ipex.xpu.has_channels_last_1d()` is deprecated.",
    category=FutureWarning,
)
def has_channels_last_1d():
    return _C._is_channels_last_1d_enabled()


@lru_cache(None)
def has_fp64_dtype(device: int = -1) -> bool:
    r"""Returns a bool indicating if the current XPU device supports dtype float64"""
    return _C._has_fp64_dtype(device)


def has_2d_block_array(device: int = 0) -> bool:
    r"""Returns a bool indicating if the platform supports 2d block array load/store"""
    return _C._has_2d_block_array(device)


def has_xmx(device: int = 0) -> bool:
    r"""Returns a bool indicating if the platform supports xmx"""
    return _C._has_xmx(device)


# Basic OnOff
class OnOff:
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


class LogLevel(EnumBase):
    DISABLED = intel_extension_for_pytorch._C.LogLevel.DISABLED
    TRACE = intel_extension_for_pytorch._C.LogLevel.TRACE
    DEBUG = intel_extension_for_pytorch._C.LogLevel.DEBUG
    INFO = intel_extension_for_pytorch._C.LogLevel.INFO
    WARN = intel_extension_for_pytorch._C.LogLevel.WARN
    ERR = intel_extension_for_pytorch._C.LogLevel.ERR
    FATAL = intel_extension_for_pytorch._C.LogLevel.FATAL


def get_log_level():
    return LogLevel.get_value(_C._get_log_level)


def set_log_level(level):
    LogLevel.set_value(_C._set_log_level, level)


def get_log_output_file_path():
    return _C._get_log_output_file_path()


def set_log_output_file_path(path):
    _C._set_log_output_file_path(path)


def get_log_rotate_file_size():
    return _C._get_log_rotate_file_size()


def set_log_rotate_file_size(size):
    assert size > 0, "Invalid file size, it should be bigger than 1mb"
    _C._set_log_rotate_file_size(size)


def get_log_split_file_size():
    return _C._get_log_split_file_size()


def set_log_split_file_size(size):
    assert size > 0, "Invalid file size, it should be bigger than 1mb"
    _C._set_log_split_file_size(size)


def get_log_component():
    return _C._get_log_component()


def set_log_component(component):
    _C._set_log_component(component)


# usage for ipex Logger, can use as torch.xpu.Logger(xxx) to set the ipex logging settings
class Logger:
    def __init__(
        self,
        level=3,
        output_file_path="",
        rotate_file_size=-1,
        split_file_size=-1,
        log_component="ALL",
    ):
        if level >= -1 and level <= 5:
            self.level = level
        else:
            raise RuntimeError(
                "Unexpected log level, need -1 to 5, but meet value {}!".format(level)
            )

        self.output_file_path = output_file_path

        if rotate_file_size > 0 and split_file_size > 0:
            raise RumtimeError(
                "Rotate file logging and split file logging can not be used ad the same time"
            )
        elif rotate_file_size > 0:
            self.rotate_file_size = rotate_file_size
            set_log_rotate_file_size(self.rotate_file_size)
        elif split_file_size > 0:
            self.split_file_size = split_file_size
            set_log_split_file_size(self.split_file_size)

        self.log_component = log_component

        set_log_level(self.level)
        set_log_output_file_path(self.output_file_path)
        set_log_component(self.log_component)

    def __enter__(self):
        if self.level != LogLevel.DISABLED:
            if rotate_file_size > 0 and split_file_size > 0:
                raise RumtimeError(
                    "Rotate file logging and split file logging can not be used ad the same time"
                )
            elif rotate_file_size > 0:
                self.rotate_file_size = rotate_file_size
                set_log_rotate_file_size(self.rotate_file_size)
            elif split_file_size > 0:
                self.split_file_size = split_file_size

            set_log_split_file_size(self.split_file_size)
            set_log_output_file_path(self.output_file_path)
            set_log_component(self.log_component)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: when ouside are using another ipex log, will cause an error, it just disable all the ipex log here
        set_log_level(LogLevel.DISABLED)
        return False


# oneDNN Verbose
class OnednnVerbLevel(EnumBase):
    OFF = 0
    ON = 1
    ON_DETAIL = 2


def set_onednn_verbose(level):
    st = OnednnVerbLevel.set_value(_C._set_onednn_verbose, level)
    assert bool(st), "WARNING: Failed to turn on oneDNN verbose!"


class onednn_verbose(object):
    def __init__(self, level):
        self.level = OnednnVerbLevel.convert(level)

    def __enter__(self):
        if self.level != OnednnVerbLevel.OFF:
            set_onednn_verbose(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_onednn_verbose(OnednnVerbLevel.OFF)
        return False


# oneMKL Verbose
class OnemklVerbLevel(EnumBase):
    OFF = 0
    ON = 1
    ON_SYNC = 2


def set_onemkl_verbose(level):
    st = OnemklVerbLevel.set_value(_C._set_onemkl_verbose, level)
    assert bool(st), "WARNING: Failed to turn on oneMKL verbose!"


class onemkl_verbose(object):
    def __init__(self, level):
        self.level = OnemklVerbLevel.convert(level)

    def __enter__(self):
        if self.level != OnemklVerbLevel.OFF:
            set_onemkl_verbose(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_onemkl_verbose(OnemklVerbLevel.OFF)
        return False


def optimize(
    model,
    dtype=None,
    optimizer=None,
    level="O1",
    inplace=False,
    conv_bn_folding=None,
    linear_bn_folding=None,
    weights_prepack=None,
    replace_dropout_with_identity=None,
    optimize_lstm=None,
    split_master_weight_for_bf16=None,
    fuse_update_step=None,
    auto_kernel_selection=None,
    sample_input=None,
    graph_mode=None,
):
    r"""
    torch.xpu.optimize is an alternative of optimize API in Intel® Extension for
    PyTorch*, to provide identical usage for XPU device only. The motivation of
    adding this alias is to unify the coding style in user scripts base on torch.xpu
    modular.

    Examples:
        >>> # bfloat16 inference case.
        >>> model = ...
        >>> model.load_state_dict(torch.load(PATH))
        >>> model.eval()
        >>> optimized_model = torch.xpu.optimize(model, dtype=torch.bfloat16)
        >>> # running evaluation step.
        >>> # bfloat16 training case.
        >>> optimizer = ...
        >>> model.train()
        >>> optimized_model, optimized_optimizer = torch.xpu.optimize(model, dtype=torch.bfloat16, optimizer=optimizer)
        >>> # running training step.
    """
    return frontend.optimize(
        model,
        dtype,
        optimizer,
        level,
        inplace,
        conv_bn_folding,
        linear_bn_folding,
        weights_prepack,
        replace_dropout_with_identity,
        optimize_lstm,
        split_master_weight_for_bf16,
        fuse_update_step,
        auto_kernel_selection,
        sample_input,
        graph_mode,
    )


class FP32MathMode(EnumBase):
    FP32 = intel_extension_for_pytorch._C.XPUFP32MathMode.FP32
    TF32 = intel_extension_for_pytorch._C.XPUFP32MathMode.TF32
    BF32 = intel_extension_for_pytorch._C.XPUFP32MathMode.BF32


def get_fp32_math_mode():
    return FP32MathMode.get_value(intel_extension_for_pytorch._C._get_fp32_math_mode)


def set_fp32_math_mode(mode):
    st = FP32MathMode.set_value(
        intel_extension_for_pytorch._C._set_fp32_math_mode, mode
    )
    assert bool(st), "WARNING: Failed to set FP32 math mode!"


class fp32_math_mode(object):
    def __init__(self, mode):
        self.mode = FP32MathMode.convert(mode)

    def __enter__(self):
        current_math_mode = get_fp32_math_mode()
        if self.mode != current_math_mode:
            set_fp32_math_mode(self.mode)
            self.mode = current_math_mode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_fp32_math_mode(self.mode)
        return False


################################################################
# EXPERIMENTAL options:
# NOTE: Below options are under experimental.
#       They are instable, and may be removed without notice!
################################################################


def has_xetla():
    return _C._is_xetla_enabled()


def get_compiler_version():
    return _C._get_compiler_version()


# oneDNN Layout
def using_onednn_layout():
    return _C._is_onednn_layout_enabled()


def is_onednn_layout(tensor):
    return torch.ops.torch_ipex.check_onednn_layout(tensor)


def enable_onednn_layout():
    _C._enable_onednn_layout()


def disable_onednn_layout():
    _C._disable_onednn_layout()


class onednn_layout(OnOff):
    def __init__(self):
        super().__init__(
            using_onednn_layout, enable_onednn_layout, disable_onednn_layout
        )


# For several primitive implementations, force to set compute engine
class XPUComputeEng(EnumBase):
    RECOMMEND = intel_extension_for_pytorch._C.XPUComputeEng.RECOMMEND
    BASIC = intel_extension_for_pytorch._C.XPUComputeEng.BASIC
    ONEDNN = intel_extension_for_pytorch._C.XPUComputeEng.ONEDNN
    ONEMKL = intel_extension_for_pytorch._C.XPUComputeEng.ONEMKL
    XETLA = intel_extension_for_pytorch._C.XPUComputeEng.XETLA


def get_compute_eng():
    return XPUComputeEng.get_value(intel_extension_for_pytorch._C._get_compute_eng)


def set_compute_eng(eng):
    st = XPUComputeEng.set_value(intel_extension_for_pytorch._C._set_compute_eng, eng)
    assert bool(st), "WARNING: Failed to set XPU compute engine!"


class compute_eng(object):
    def __init__(self, eng):
        self.eng = XPUComputeEng.convert(eng)

    def __enter__(self):
        current_compute_eng = get_compute_eng()
        if self.eng != current_compute_eng:
            set_compute_eng(self.eng)
            self.eng = current_compute_eng
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_compute_eng(self.eng)
        return False
