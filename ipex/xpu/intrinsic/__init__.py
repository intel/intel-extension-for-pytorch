from .modules import LinearReLU
from .modules import LinearSigmoid
from .modules import ReLUDummy

import ipex

__all__ = [
    'LinearReLU',
    'LinearSigmoid',
    'ReLUDummy',
]


def MulAdd(input, other, accumu, alpha=1.0):
    return ipex._C.mul_add(input, other, accumu, alpha)
