import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core

F_adaptive_avg_pool2d = F.adaptive_avg_pool2d
torch_max_pool2d = torch.max_pool2d
torch_max_pool3d = torch.max_pool3d

class AdaptiveAvgPool2dFunction(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        _output_size = _list_with_default(output_size, input.size())
        output = core.adaptive_avg_pool2d(input, _output_size)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = core.adaptive_avg_pool2d_backward(grad_output, input)
        return (grad_input, None)

class MaxPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, ceil_mode):
        output = core.max_pooling(input, (kernel_size,), (stride,), (padding,), (dilation,), ceil_mode)
        ctx.save_for_backward(output, input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input= ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = core.max_pooling_backward(grad_output, output, input, (ctx.kernel_size,), (ctx.stride,), (ctx.padding,), (ctx.dilation,), ctx.ceil_mode)
        return (grad_input, None, None, None, None, None)

def _list_with_default(out_size, defaults):
    if isinstance(out_size, int):
        return (out_size,)
    if len(defaults) <= len(out_size):
        raise ValueError('Input dimension should be at least {}'.format(len(out_size) + 1))
    return [v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size):])]

def adaptive_avg_pool2d(input, output_size):
    if input.device.type == 'dpcpp':
        return AdaptiveAvgPool2dFunction.apply(input, output_size)
    return F_adaptive_avg_pool2d(input, output_size)

def max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode):
    if input.device.type == 'dpcpp':
        return MaxPoolingFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    return torch_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

def max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode):
    if input.device.type == 'dpcpp':
        return MaxPoolingFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    return torch_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)

F.adaptive_avg_pool2d = adaptive_avg_pool2d
torch.max_pool2d = max_pool2d
torch.max_pool3d = max_pool3d