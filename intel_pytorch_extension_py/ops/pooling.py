import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core
from torch.nn.modules.utils import _single, _pair
from typing import List

Vector = List[int]

def adaptive_avg_pool2d(input, output_size: Vector):
    return torch.ops.torch_ipex.adaptive_avg_pool2d(input, _pair(output_size))

def max_pool3d(input, kernel_size: Vector, stride: Vector, padding: Vector, dilation: Vector, ceil_mode: bool):
    if len(_single(stride)) == 0:
        stride = kernel_size
    return torch.ops.torch_ipex.max_pool3d(input, _single(kernel_size), _single(stride), _single(padding), _single(dilation), ceil_mode)

def max_pool2d(input, kernel_size: Vector, stride: Vector, padding: Vector, dilation: Vector, ceil_mode: bool):
    if len(_pair(stride)) == 0:
        stride = kernel_size
    return torch.ops.torch_ipex.max_pool2d(input, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), ceil_mode)

torch.adaptive_avg_pool2d = adaptive_avg_pool2d
torch.max_pool2d = max_pool2d
torch.max_pool3d = max_pool3d
