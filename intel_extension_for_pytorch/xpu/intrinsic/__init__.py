import torch
from .modules import Interaction
import intel_extension_for_pytorch

__all__ = [
    'Interaction',
]


def MulAdd(input, other, accumu, alpha=1.0):
    return torch.ops.torch_ipex.mul_add(input, other, accumu, alpha)
