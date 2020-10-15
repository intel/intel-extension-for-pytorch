import os
import math

import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.nn import init
from .lib import torch_ipex
from scripts.version import __version__

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
        return torch_ipex.linear_relu(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def mul_add(input, other, accumu, alpha=1.0):
    return torch_ipex.mul_add(input, other, accumu, alpha)


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

def _usm_pstl_is_enabled():
    return torch_ipex._usm_pstl_is_enabled()

def _double_kernel_disabled():
    return torch_ipex._double_kernel_disabled()

def _onemkl_is_enabled():
    return torch_ipex._onemkl_is_enabled()
