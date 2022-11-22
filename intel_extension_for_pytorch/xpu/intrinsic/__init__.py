import torch
from .modules import Interaction
import intel_extension_for_pytorch

__all__ = [
    'Interaction',
    'nms',
]


def MulAdd(input, other, accumu, alpha=1.0):
    return torch.ops.torch_ipex.mul_add(input, other, accumu, alpha)


def nms(dets, scores, iou_threshold):
    return torch.ops.torch_ipex.nms(dets, scores, iou_threshold)
