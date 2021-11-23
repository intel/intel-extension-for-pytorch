import contextlib
import sys
from typing import Optional, Tuple, Union
import torch
import ipex
from torch import device as _device

from .streams import Stream
from .random import *
from .memory import *
from .intrinsic import *
from .settings import *

from ipex._utils import _get_device_index  # , _dummy_type
import ipex.optim as optim
import ipex.autograd as autograd
from ipex.autograd import inference_mode

default_generators: Tuple[torch._C.Generator] = ipex._C.default_generators
_device_t = Union[_device, str, int]


def _lazy_init():
    pass


def is_available() -> bool:
    r"""Returns a bool indicating if XPU is currently available."""
    # if not hasattr(torch._C, '_xpu_getDeviceCount'):
    #     return False
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    # return _C._xpu_getDeviceCount() > 0
    return True


def device_count() -> int:
    r"""Returns the number of XPUs device available."""
    if is_available():
        return ipex._C._getDeviceCount()
    else:
        return 0


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
        self.prev_idx = ipex._C._getDevice()
        if self.prev_idx != self.idx:
            ipex._C._setDevice(self.idx)
        _lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            ipex._C._setDevice(self.prev_idx)
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
        ipex._C._setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Gets the xpu capability of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor xpu capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: _device_t):
    # _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return ipex._C._get_device_properties(device)


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    _lazy_init()
    return ipex._C._getDevice()


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a XPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    idx = _get_device_index(device, optional=True)
    return ipex._C._synchronize(idx)


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return Stream(_cdata=ipex._C._getCurrentStream(
        _get_device_index(device, optional=True)))


@contextlib.contextmanager
def stream(stream):
    r"""Context-manager that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    if stream is None:
        yield
        return
    src_prev_stream = current_stream()

    if src_prev_stream.device != stream.device:
        # The given stream is on a different device; have to restore the
        # current_stream on that device on exit as well
        with device(stream.device):
            dst_prev_stream = current_stream()

    ipex._C._setCurrentStream(stream._cdata)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            ipex._C._setCurrentStream(dst_prev_stream._cdata)
        ipex._C._setCurrentStream(src_prev_stream._cdata)


from torch.storage import _StorageBase


class IntStorage(ipex._C.IntStorageBase, _StorageBase):
    pass


class LongStorage(ipex._C.LongStorageBase, _StorageBase):
    pass


class BoolStorage(ipex._C.BoolStorageBase, _StorageBase):
    pass


class HalfStorage(ipex._C.HalfStorageBase, _StorageBase):
    pass


class DoubleStorage(ipex._C.DoubleStorageBase, _StorageBase):
    pass


class FloatStorage(ipex._C.FloatStorageBase, _StorageBase):
    pass


class BFloat16Storage(ipex._C.BFloat16StorageBase, _StorageBase):
    pass


torch._storage_classes.add(IntStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(BFloat16Storage)
ipex._C._initExtension()


def _xpu_tag(obj):
    if type(obj).__module__ == 'ipex.xpu':
        return 'xpu:' + str(obj.get_device())


def validate_xpu_device(location):
    # device = torch.xpu._utils._get_device_index(location, True)
    #
    # if not torch.xpu.is_available():
    #     raise RuntimeError('Attempting to deserialize object on a xpu '
    #                        'device but torch.xpu.is_available() is False. '
    #                        'If you are running on a CPU-only machine, '
    #                        'please use torch.load with map_location=torch.device(\'cpu\') '
    #                        'to map your storages to the CPU.')
    # device_count = torch.xpu.device_count()
    # if device >= device_count:
    #     raise RuntimeError('Attempting to deserialize object on xpu device '
    #                        f'{device} but torch.xpu.device_count() is {device_count}. Please use '
    #                        'torch.load with map_location to map your storages '
    #                        'to an existing device.')
    # return device
    return current_device()


current_module = sys.modules[__name__]


def _xpu(self, device_idx=None, non_blocking=False, **kwargs):
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
    # non_blocking = _get_async_or_non_blocking('xpu', non_blocking, kwargs)
    # if self.is_xpu:
    #     if device is None:
    #         device = torch.xpu.current_device()
    #     if self.get_device() == device:
    #         return self
    # else:
    #     if device is None:
    #         device = -1
    with device(device_idx):
        if self.is_sparse:
            # new_type = getattr(torch.xpu.sparse, self.__class__.__name__)
            # indices = torch._indices(self).xpu(device, non_blocking)
            # values = torch._values(self).xpu(device, non_blocking)
            # return new_type(indices, values, self.size())
            pass
        else:
            new_type = getattr(current_module, self.__class__.__name__)
            return new_type(self.size()).copy_(self, non_blocking)


def _xpu_deserialize(obj, location):
    if location.startswith('xpu'):
        device_id = validate_xpu_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            storage_type = getattr(current_module, type(obj).__name__)
            with device(device_id):
                return storage_type(obj.size())
        else:
            return _xpu(obj, device=device_id)


def get_device_type() -> str:
    return 'xpu'


from torch import serialization

serialization.register_package(30, _xpu_tag, _xpu_deserialize)

torch._register_device_module('xpu', current_module)

# post initial
ipex._C._postInitExtension()

del torch
