import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core
from torch.nn.modules.utils import _single, _pair
from typing import List

Vector = List[int]

torch_max_pool3d = torch.max_pool3d

class MaxPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, ceil_mode):
        ctx.kernel_size = _single(kernel_size)
        ctx.stride = _single(stride)
        ctx.padding = _single(padding)
        ctx.dilation = _single(dilation)
        ctx.ceil_mode = ceil_mode
        output = core.max_pooling(input, ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation, ctx.ceil_mode)
        ctx.save_for_backward(output, input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input= ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = core.max_pooling_backward(grad_output, output, input, ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation, ctx.ceil_mode)
        return (grad_input, None, None, None, None, None)

def adaptive_avg_pool2d(input, output_size: Vector):
    return torch.ops.torch_ipex.adaptive_avg_pool2d(input, _pair(output_size))

def max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode):
    try:
        if input.device.type == 'dpcpp' and core.get_auto_dnnl():
            return MaxPoolingFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    except RuntimeError:
        pass
    return torch_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)

def max_pool2d(input, kernel_size: Vector, stride: Vector, padding: Vector, dilation: Vector, ceil_mode: bool):
    if not stride:
        stride = kernel_size
    return torch.ops.torch_ipex.max_pool2d(input, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), ceil_mode)

torch._C._nn.adaptive_avg_pool2d = adaptive_avg_pool2d
torch.max_pool2d = max_pool2d
torch.max_pool3d = max_pool3d
