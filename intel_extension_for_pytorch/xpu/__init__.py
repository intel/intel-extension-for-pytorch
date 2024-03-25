# -*- coding: utf-8 -*-
r"""
This package is lazily initialized, so you can always import it.
"""
import ctypes
from functools import lru_cache
import sys
from typing import List, Optional, Tuple, Union, Dict

import torch
import intel_extension_for_pytorch

from torch import serialization
from torch.storage import _StorageBase, _LegacyStorage, _warn_typed_storage_removal
from torch import device as _device
from torch._utils import classproperty
from torch.xpu._utils import _get_device_index

from .lazy_init import (
    _lazy_init,
    _lazy_call,
    _is_initialized,
    is_initialized,
    _is_in_bad_fork,
)
from .streams import Stream, Event
from .intrinsic import *
from .cpp_extension import *
from .amp import *
from .utils import *
from .random import *
from .deterministics import *
from .memory import *
from ..utils.channels_last_1d import is_contiguous_channels_last_1d, to_channels_last_1d
from ..utils.capsule import get_pointer_from_capsule
from ..utils.utils import has_xpu

from .overrides import (
    override_tensor_totype,
    override_assert_equal,
)

import intel_extension_for_pytorch.optim as optim
from intel_extension_for_pytorch._version import (
    __version__,
    __ipex_gitrev__,
    __torch_gitrev__,
    __gpu_onednn_gitrev__,
    __build_type__,
)

default_generators: Tuple[torch._C.Generator] = ()
_device_t = Union[_device, str, int]


def init():
    r"""Initialize the XPU's state. This is a Python API about lazy initialization
    that avoids initializing XPU until the first time it is accessed. You may need
    to call this function explicitly in very rare cases, since IPEX could call
    this initialization automatically when XPU functionality is on-demand.

    Does nothing if call this function repeatedly.
    """
    _lazy_init()


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a XPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    idx = _get_device_index(device, optional=True)
    return intel_extension_for_pytorch._C._synchronize(idx)


class StreamContext(object):
    r"""Context-manager that selects a given stream.

    All XPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["Stream"]

    def __init__(self, stream: Optional["Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = None
        self.dst_prev_stream = None

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or XPU device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = current_stream(cur_stream.device)
        set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no XPU device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:
            set_stream(self.dst_prev_stream)
        set_stream(self.src_prev_stream)


def stream(stream: Optional["Stream"]) -> StreamContext:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and device type

    Args:
        stream_id (int): not visible to the user, used to assigned to the
            specific stream.
        device_index (int): selected device index.
        device_type (int): selected device type.
    """
    intel_extension_for_pytorch._C._setCurrentStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = intel_extension_for_pytorch._C._getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the XPU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """

    idx = device.index
    if idx is None:
        idx = torch.xpu.current_device()
    return torch.xpu.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: Union[int, str, torch.device] = "xpu"
) -> None:
    r"""Sets the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: Union[int, str, torch.device] = "xpu") -> int:
    r"""Returns the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _XPUBase.__new__
    return super(_XPUBase, cls).__new__(cls, *args, **kwargs)


class _XPUBase(object):
    is_xpu = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super(_XPUBase, self).type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new


class _XPULegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for XPU storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for XPU storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for XPU storage")


class ByteStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat


del _LegacyStorage
del _XPULegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)


def _xpu_tag(obj):
    if obj.device.type == "xpu":
        return "xpu:" + str(obj.device.index)


def validate_xpu_device(location):
    device = _get_device_index(location, True)
    if not torch.xpu.is_available():
        raise RuntimeError(
            "Attempting to deserialize object on a xpu "
            "device but torch.xpu.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    device_count = torch.xpu.device_count()
    if device >= device_count:
        raise RuntimeError(
            "Attempting to deserialize object on xpu device "
            f"{device} but torch.xpu.device_count() is {device_count}. Please use "
            "torch.load with map_location to map your storages "
            "to an existing device."
        )
    return device


current_module = sys.modules[__name__]


def _xpu(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in xpu memory.

    If this object is already in xpu memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    non_blocking = torch._utils._get_async_or_non_blocking("xpu", non_blocking, kwargs)
    # if self.is_xpu:
    #     if device is None:
    #         device = torch.xpu.current_device()
    #     if self.get_device() == device:
    #         return self
    # else:
    if device is None:
        device = -1
    with torch.xpu.device(device):
        if self.is_sparse:
            # new_type = getattr(torch.xpu.sparse, self.__class__.__name__)
            # indices = torch._indices(self).xpu(device, non_blocking)
            # values = torch._values(self).xpu(device, non_blocking)
            # return new_type(indices, values, self.size())
            pass
        else:
            untyped_storage = torch.UntypedStorage(
                self.size(), device=torch.device("xpu")
            )
            untyped_storage.copy_(self, non_blocking)
            return untyped_storage


def _xpu_deserialize(obj, location):
    if location.startswith("xpu"):
        device = validate_xpu_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            with torch.xpu.device(device):
                return torch.UntypedStorage(obj.nbytes(), device=torch.device(location))
        else:
            return _xpu(obj, device=device)


def _register_torch_device_module(device_type, module):
    device_type = torch.device(device_type).type
    torch_module = sys.modules["torch"]
    registered_module = getattr(torch_module, device_type, None)
    if registered_module:
        for sub_module_key in dir(registered_module):
            if not hasattr(module, sub_module_key):
                setattr(
                    module,
                    sub_module_key,
                    getattr(registered_module, sub_module_key),
                )
        setattr(torch_module, device_type, module)
        torch_module_name = ".".join(["torch", device_type])
        sys.modules[torch_module_name] = module
    else:
        torch._register_device_module(device_type, module)


if utils.has_xpu():
    _StorageBase.xpu = _xpu

    serialization.register_package(30, _xpu_tag, _xpu_deserialize)

    _register_torch_device_module("xpu", current_module)

    # post initial
    intel_extension_for_pytorch._C._postInitExtension()

    override_tensor_totype()
    exec_path = sys.argv[0].split("/")
    if len(exec_path) > 0 and "pytest" in exec_path:
        override_assert_equal()


# XXX: this is a temporary work-around to replace torch's _prepare_profiler method
#     inside IPEX because the extension need to prepare its profiler as a path.
#     When IPEX's PTI based profiler is successfully upstream to PyTorch as the
#     Kineto's plugin, these part should be removed as well as all profiler code
#     inside IPEX.

torch_prepare_profiler_method = torch.autograd.profiler._prepare_profiler


def _prepare_profiler(config, activities):
    if torch.profiler.ProfilerActivity.XPU not in activities:
        return torch_prepare_profiler_method(config, activities)
    else:
        if intel_extension_for_pytorch._C._is_pti_enabled():
            return intel_extension_for_pytorch._C._prepare_profiler(config, activities)
        else:
            raise RuntimeError(
                "intel_extension_for_pytorch ot build with PTI support. "
                "Cannot profile on XPU activities, but has set ProfilerActivity.XPU"
            )
            return None


if "torch.autograd.profiler" in sys.modules:
    mod = sys.modules["torch.autograd.profiler"]
    mod._prepare_profiler = _prepare_profiler
else:
    import torch.autograd.profiler

    mod = sys.modules["torch.autograd.profiler"]
    mod._prepare_profiler = _prepare_profiler
