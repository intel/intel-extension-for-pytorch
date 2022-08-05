r"""
This package is lazily initialized, so you can always import it.
"""

import contextlib
import sys
from typing import List, Optional, Tuple, Union
import torch
import intel_extension_for_pytorch
from torch import device as _device
import traceback
import threading

from .streams import Stream, Event
from .intrinsic import *
from .settings import *
from .itt import emit_itt
from .amp import *
from .utils import *

from intel_extension_for_pytorch._utils import _get_device_index  # , _dummy_type
import intel_extension_for_pytorch.optim as optim
import intel_extension_for_pytorch.autograd as autograd
from intel_extension_for_pytorch.autograd import inference_mode

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # invoke these after initialization occurs
_is_in_bad_fork = getattr(intel_extension_for_pytorch._C, "_xpu_isInBadFork", lambda: False)


class _LazySeedTracker:
    # We only track the latest seed given by 'manual_seed_all' or 'manual_seed'.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = [] 

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> List:
        return self.call_order


_lazy_seed_tracker = _LazySeedTracker()

default_generators: Tuple[torch._C.Generator] = ()
_device_t = Union[_device, str, int]

def to_channels_last_1d(t):
    if isinstance(t, torch.nn.Module):
        for m in t.modules():
            for param in m.parameters():
                if isinstance(m, (torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.MaxPool1d)):
                    if 3 == param.data.dim():
                        if 1 == param.data.size(0):
                            param.data = param.data.transpose(1, -1).contiguous().transpose(1, -1)
                        elif 1 == param.data.size(1):
                            param.data = param.data.as_strided(
                                param.data.size(), (param.data.size(1) * param.data.size(-1), 1, param.data.size(1)))
                        else:
                            param.data = param.data.view(
                                param.data.size(0), param.data.size(1), 1, param.data.size(2))
                            param.data = param.data.to(
                                memory_format=torch.channels_last)
                            param.data = param.data.view(
                                param.data.size(0), param.data.size(1), param.data.size(3))
        return t

    if 3 == t.dim():
        if 1 == t.size(0):
            t = t.transpose(1, -1).contiguous().transpose(1, -1)
        elif 1 == t.size(1):
            t = t.as_strided(t.size(), (t.size(1) * t.size(-1), 1, t.size(1)))
        else:
            t = t.view(t.size(0), t.size(1), 1, t.size(2))
            t = t.to(memory_format=torch.channels_last)
            t = t.view(t.size(0), t.size(1), t.size(3))
    return t

def is_contiguous_channels_last_1d(input):
    if 3 != input.dim():
        return False

    tmpTen = input.view(input.size(0), input.size(1), 1, input.size(2))
    if tmpTen.is_contiguous(memory_format=torch.channels_last):
        return True
    else:
        return False

def is_initialized():
    r"""Returns whether XPU state has been initialized."""
    return _initialized and not _is_in_bad_fork()


class DeferredXPUCallError(Exception):
    pass


def init():
    r"""Initialize the XPU's state. This is a Python API about lazy initialization
    that avoids initializing XPU until the first time it is accessed. You may need 
    to call this function explicitly in very rare cases, since IPEX could call
    this initialization automatically when XPU functionality is on-demand.

    Does nothing if call this function repeatedly.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # This test was was protected via GIL. Double-check whether XPU has
        # already been initialized. If a thread acquired the lock first,
        # it will do an initialization. When the other threads get the lock,
        # they will find XPU has been initialized.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL.
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize XPU in forked subprocess. To use XPU with "
                "multiprocessing, you must use the 'spawn' start method")
        if not hasattr(intel_extension_for_pytorch._C, '_getDeviceCount'):
            raise AssertionError("IPEX not compiled with XPU enabled")
        # This function detects bad fork processing and throws if there's a device
        # initialization error, no XPUs are found or any other error occurs
        intel_extension_for_pytorch._C._initExtension() 
        # Some of the queued calls in _queued_calls[] may reentrantly call 
        # _lazy_init(). We must prevent multiple initializations. In that case
        # just return early without initializeing to avoid a deadlock.
        _tls.is_initializing = True

        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (f"XPU call failed lazily at initialization with error: {str(e)}\n\n"
                           f"XPU call was originally invoked at:\n\n{orig_traceback}")
                    raise DeferredXPUCallError(msg) from e
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True


def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))


def is_available() -> bool:
    r"""Returns a bool indicating if XPU is currently available."""
    if not hasattr(intel_extension_for_pytorch._C, '_getDeviceCount'):
        return False
    # This function never throws and returns 0 if driver is missing or can't
    # be initialized
    return intel_extension_for_pytorch._C._getDeviceCount() > 0


def device_count() -> int:
    r"""Returns the number of XPUs device available."""
    if is_available():
        return intel_extension_for_pytorch._C._getDeviceCount()
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


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return Stream(_cdata=intel_extension_for_pytorch._C._getCurrentStream(
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

    intel_extension_for_pytorch._C._setCurrentStream(stream._cdata)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            intel_extension_for_pytorch._C._setCurrentStream(dst_prev_stream._cdata)
        intel_extension_for_pytorch._C._setCurrentStream(src_prev_stream._cdata)


from .random import *

from .memory import *

from torch.storage import _StorageBase

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


class ShortStorage(_XPUBase, intel_extension_for_pytorch._C.ShortStorageBase, _StorageBase):
    pass


class CharStorage(_XPUBase, intel_extension_for_pytorch._C.CharStorageBase, _StorageBase):
    pass


class IntStorage(_XPUBase, intel_extension_for_pytorch._C.IntStorageBase, _StorageBase):
    pass


class LongStorage(_XPUBase, intel_extension_for_pytorch._C.LongStorageBase, _StorageBase):
    pass


class BoolStorage(_XPUBase, intel_extension_for_pytorch._C.BoolStorageBase, _StorageBase):
    pass


class HalfStorage(_XPUBase, intel_extension_for_pytorch._C.HalfStorageBase, _StorageBase):
    pass


class DoubleStorage(_XPUBase, intel_extension_for_pytorch._C.DoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_XPUBase, intel_extension_for_pytorch._C.FloatStorageBase, _StorageBase):
    pass


class BFloat16Storage(_XPUBase, intel_extension_for_pytorch._C.BFloat16StorageBase, _StorageBase):
    pass


class QUInt8Storage(_XPUBase, intel_extension_for_pytorch._C.QUInt8StorageBase, _StorageBase):
    pass


class QInt8Storage(_XPUBase, intel_extension_for_pytorch._C.QInt8StorageBase, _StorageBase):
    pass


torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(QUInt8Storage)
torch._storage_classes.add(QInt8Storage)


def _xpu_tag(obj):
    if type(obj).__module__ == 'intel_extension_for_pytorch.xpu':
        return 'xpu:' + str(obj.get_device())


def validate_xpu_device(location):
    device = _get_device_index(location, True)
    if not torch.xpu.is_available():
        raise RuntimeError('Attempting to deserialize object on a xpu '
                           'device but torch.xpu.is_available() is False. '
                           'If you are running on a CPU-only machine, '
                           'please use torch.load with map_location=torch.device(\'cpu\') '
                           'to map your storages to the CPU.')
    device_count = torch.xpu.device_count()
    if device >= device_count:
        raise RuntimeError('Attempting to deserialize object on xpu device '
                           f'{device} but torch.xpu.device_count() is {device_count}. Please use '
                           'torch.load with map_location to map your storages '
                           'to an existing device.')
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
    non_blocking = torch._utils._get_async_or_non_blocking('xpu', non_blocking, kwargs)
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
intel_extension_for_pytorch._C._postInitExtension()
