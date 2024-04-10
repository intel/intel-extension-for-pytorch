r"""
This package is lazily initialized, so you can always import it.
"""

from torch import serialization
from torch.storage import _StorageBase
import sys
from typing import List, Optional, Tuple, Union, Dict
import torch
import intel_extension_for_pytorch
from .lazy_init import _lazy_init, _lazy_call, _is_in_bad_fork
from torch import device as _device
from torch._utils import classproperty

from ._proxy_module import *
from .streams import Stream, Event
from .intrinsic import *
from .cpp_extension import *
from .amp import *
from .utils import *
from .random import *
from .memory import *
from ..utils.channels_last_1d import is_contiguous_channels_last_1d, to_channels_last_1d

from .overrides import (
    override_tensor_totype,
    override_assert_equal,
    override_get_stream,
    override_recursive_to,
)

from .generator import Generator

from torch._utils import _get_device_index
import intel_extension_for_pytorch.optim as optim
from intel_extension_for_pytorch._version import (
    __version__,
    __ipex_gitrev__,
    __torch_gitrev__,
    __gpu_onednn_gitrev__,
    __build_type__,
)  # noqa B950

default_generators: Tuple[torch._C.Generator] = ()
_device_t = Union[_device, str, int]


def is_initialized():
    r"""Returns whether XPU state has been initialized."""
    from .lazy_init import _initialized

    return _initialized


def init():
    r"""Initialize the XPU's state. This is a Python API about lazy initialization
    that avoids initializing XPU until the first time it is accessed. You may need
    to call this function explicitly in very rare cases, since IPEX could call
    this initialization automatically when XPU functionality is on-demand.

    Does nothing if call this function repeatedly.
    """
    _lazy_init()


# This API call _prefetchDeviceCount() if _lazy_init() has not been called such that
# this API can be used before forking proces.
def device_count() -> int:
    r"""Returns the number of XPUs device available."""
    if hasattr(intel_extension_for_pytorch._C, "_getDeviceCount"):
        if is_initialized():
            return intel_extension_for_pytorch._C._getDeviceCount()
        else:
            return intel_extension_for_pytorch._C._prefetchDeviceCount()
    else:
        return 0


# This API can be used before forking process if _lazy_init() has not been called.
def is_available() -> bool:
    r"""Returns a bool indicating if XPU is currently available."""
    # This function device_count() never throws and returns 0 if driver is missing
    # or can't be initialized
    return device_count() > 0


# This API can be used before forking process if _lazy_init() has not been called.
def getDeviceIdListForCard(card_id=-1) -> list:
    r"""Returns the device list of card_id.
    By default, return device list of the card which contains max number of devices."""
    if hasattr(intel_extension_for_pytorch._C, "_getDeviceIdListForCard"):
        if is_initialized():
            return intel_extension_for_pytorch._C._getDeviceIdListForCard(card_id)
        else:
            return intel_extension_for_pytorch._C._prefetchDeviceIdListForCard(card_id)
    else:
        return []


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = intel_extension_for_pytorch._C._getDevice()
        if self.prev_idx != self.idx:
            intel_extension_for_pytorch._C._setDevice(self.idx)
        if not torch.jit.is_scripting():
            _lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            intel_extension_for_pytorch._C._setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_xpu else -1
        super(device_of, self).__init__(idx)


def set_device(device: _device_t) -> None:
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``xpu_VISIBLE_DEVICES`` environmental variable.

    Arguments:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        intel_extension_for_pytorch._C._setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> Dict[str, Any]:
    r"""Gets the xpu capability of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        Dict[str, Any]: the xpu capability dictionary of the device
    """
    prop = get_device_properties(device)
    return {
        "max_work_group_size": prop.max_work_group_size,
        "max_num_sub_groups": prop.max_num_sub_groups,
        "sub_group_sizes": prop.sub_group_sizes,
    }


def get_device_properties(device: _device_t):
    r"""Gets the xpu properties of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            device properties. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        _DeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return intel_extension_for_pytorch._C._get_device_properties(device)


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    # lazy initialization occurs in _getDevice
    return intel_extension_for_pytorch._C._getDevice()


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
    intel_extension_for_pytorch._C._setCurrentStream(stream._cdata)


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return Stream(
        _cdata=intel_extension_for_pytorch._C._getCurrentStream(
            _get_device_index(device, optional=True)
        )
    )


from torch.storage import _LegacyStorage


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
        return torch.uint8


class DoubleStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.double


class FloatStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.float


class HalfStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.half


class LongStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.long


class IntStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.int


class ShortStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.short


class CharStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.int8


class BoolStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.bool


class BFloat16Storage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_XPULegacyStorage):
    @classproperty
    def dtype(self):
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
        device_id = validate_xpu_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            with torch.xpu.device(device):
                return torch.UntypedStorage(obj.nbytes(), device=torch.device(location))
        else:
            return _xpu(obj, device=device_id)


def get_device_type() -> str:
    return "xpu"


if intel_extension_for_pytorch._C._has_xpu():
    _StorageBase.xpu = _xpu
    serialization.register_package(30, _xpu_tag, _xpu_deserialize)
    torch._register_device_module("xpu", current_module)
    intel_extension_for_pytorch._C._postInitExtension()
    if is_available():
        override_get_stream()
        override_recursive_to()
        if not has_fp64_dtype():
            override_tensor_totype()
            exec_path = sys.argv[0].split("/")
            if len(exec_path) > 0 and "pytest" in exec_path:
                override_assert_equal()
