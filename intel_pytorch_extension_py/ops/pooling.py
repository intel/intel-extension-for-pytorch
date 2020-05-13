import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core
from torch.nn.modules.utils import _single

torch_adaptive_avg_pool2d = torch._C._nn.adaptive_avg_pool2d
torch_max_pool2d = torch.max_pool2d
torch_max_pool3d = torch.max_pool3d

class AdaptiveAvgPool2dFunction(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        output = core.adaptive_avg_pool2d(input, _single(output_size))
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

def adaptive_avg_pool2d(input, output_size):
    try:
        if input.device.type == 'dpcpp' and core.get_auto_dnnl():
            return AdaptiveAvgPool2dFunction.apply(input, output_size)
    except RuntimeError:
        pass
    return torch_adaptive_avg_pool2d(input, output_size)

def max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode):
    try:
        if input.device.type == 'dpcpp' and core.get_auto_dnnl():
            return MaxPoolingFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    except RuntimeError:
        pass
    return torch_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

def max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode):
    try:
        if input.device.type == 'dpcpp' and core.get_auto_dnnl():
            return MaxPoolingFunction.apply(input, kernel_size, stride, padding, dilation, ceil_mode)
    except RuntimeError:
        pass
    return torch_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)

torch._C._nn.adaptive_avg_pool2d = adaptive_avg_pool2d
torch.max_pool2d = max_pool2d
torch.max_pool3d = max_pool3d