from .modules import LinearReLU
from .modules import LinearSigmoid
from .modules import ReLUDummy
from .modules import Interaction

import intel_extension_for_pytorch

__all__ = [
    'LinearReLU',
    'LinearSigmoid',
    'ReLUDummy',
    'Interaction',
]


def MulAdd(input, other, accumu, alpha=1.0):
    return intel_extension_for_pytorch._C.mul_add(input, other, accumu, alpha)
