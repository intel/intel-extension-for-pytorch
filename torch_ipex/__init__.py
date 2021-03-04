import os
import math
import sys

import torch
from torch._utils import _get_async_or_non_blocking
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import init
from torch.optim.optimizer import Optimizer, required
from torch import device as _device
from ._utils import _get_device_index #, _dummy_type
from typing import List, Optional, Tuple, Union
from . import _C
from .version import __version__, __ipex_gitrev__
from .streams import Stream, Event

from . import profiler as profiler
from . import itt as itt


default_generators: Tuple[torch._C.Generator] = _C.default_generators
_device_t = Union[_device, str, int]


def version():
    print("ipex gpu version: {}".format(__version__))
    print("ipex gpu git sha: {}".format(__ipex_gitrev__))


# Customized operators:
# for now, we don't support bwk propagation
class LinearReLU(Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return _C.linear_relu(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LinearSigmoid(Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearSigmoid, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return _C.linear_sigmoid(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class SplitSGD(Optimizer):
    r"""Implements low precision stochastic gradient descent with extra state."""

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SplitSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SplitSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p.dtype == torch.bfloat16:
                    param_state = self.state[p]
                    if 'bottom_half' not in param_state:
                        b_d = param_state['bottom_half'] = torch.zeros_like(
                                p.data, dtype=torch.bfloat16, device=p.device)
                    else:
                        b_d = param_state['bottom_half']

                if p.dtype == torch.bfloat16:
                    _C.packed_add(p.data, b_d, d_p, -group['lr'])
                    param_state['bottom_half'] = b_d
                else:
                    p.data.add_(d_p, alpha=-group['lr'])
        return loss

def MulAdd(input, other, accumu, alpha=1.0):
    return _C.mul_add(input, other, accumu, alpha)


class ReLUDummy(Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLUDummy, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return input

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


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


def _find_dpcpp_home():
    pass


def _here_paths():
    here = os.path.abspath(__file__)
    torch_ipex_path = os.path.dirname(here)
    return torch_ipex_path


def include_paths():
    '''
    Get the include paths required to build a C++ extension.

    Returns:
        A list of include path strings.
    '''
    torch_ipex_path = _here_paths()
    lib_include = os.path.join(torch_ipex_path, 'include')
    paths = [
        lib_include,
        # os.path.join(lib_include, 'more')
    ]
    return paths


def library_paths():
    '''
    Get the library paths required to build a C++ extension.

    Returns:
        A list of library path strings.
    '''
    torch_ipex_path = _here_paths()
    lib_path = os.path.join(torch_ipex_path, 'lib')
    paths = [
        lib_path,
    ]
    return paths


def _onedpl_is_enabled():
    return _C._onedpl_is_enabled()

def _onemkl_is_enabled():
    return _C._onemkl_is_enabled()

def _double_kernel_disabled():
    return _C._double_kernel_disabled()


def device_count() -> int:
    r"""Returns the number of XPUs device available."""
    if is_available():
        return _C._getDeviceCount()
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
        self.prev_idx = _C._getDevice()
        if self.prev_idx != self.idx:
            _C._setDevice(self.idx)
        _lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            _C._setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """
    pass
    # def __init__(self, obj):
    #     idx = obj.get_device() if obj.is_xpu else -1
    #     super(device_of, self).__init__(idx)


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
        _C._setDevice(device)


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


def get_device_properties(device: _device_t):# -> _xpuDeviceProperties:
    # _lazy_init()  # will define _get_device_properties
    # device = _get_device_index(device, optional=True)
    # if device < 0 or device >= device_count():
    #     raise AssertionError("Invalid device id")
    # return _get_device_properties(device)
    pass


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    _lazy_init()
    return _C._getDevice()


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a XPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    pass


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return Stream(_cdata=_C._getCurrentStream(
        _get_device_index(device, optional=True)))


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    # _lazy_init()
    # return Stream(_cdata=torch._C._xpu_getDefaultStream(
    #     _get_device_index(device, optional=True)))
    pass


from torch.storage import _StorageBase


class IntStorage(_C.IntStorageBase, _StorageBase):
    pass


class LongStorage(_C.LongStorageBase, _StorageBase):
    pass


class BoolStorage(_C.BoolStorageBase, _StorageBase):
    pass


class HalfStorage(_C.HalfStorageBase, _StorageBase):
    pass


class DoubleStorage(_C.DoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_C.FloatStorageBase, _StorageBase):
    pass


class BFloat16Storage(_C.BFloat16StorageBase, _StorageBase):
    pass


torch._storage_classes.add(IntStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(BFloat16Storage)
_C._initExtension()



def _xpu_tag(obj):
    if type(obj).__module__ == 'torch_ipex':
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
        device = validate_xpu_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            storage_type = getattr(_C, type(obj).__name__)
            with _C.device(device):
                return storage_type(obj.size())
        else:
            return _xpu(obj, device=device)


def get_device_type() -> str:
    return 'xpu'


from .random import *


from torch import serialization

serialization.register_package(30, _xpu_tag, _xpu_deserialize)


torch.runtime.register_runtime('xpu', current_module)


_C._postInitExtension()

