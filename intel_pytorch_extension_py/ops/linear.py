import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core

F_linear = F.linear

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = core.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if bias == None:
            output_mask = (input.requires_grad, weight.requires_grad, 0)
        else:
            output_mask = (input.requires_grad, weight.requires_grad, bias.requires_grad)
        grad_input, grad_weight, grad_bias = core.linear_backward(input, grad_output, weight, output_mask)
        return (grad_input, grad_weight, grad_bias)

def linear(input, weight, bias=None):
    if input.device.type == 'dpcpp':
        return LinearFunction.apply(input, weight, bias)
    return F_linear(input, weight, bias)

F.linear = linear
