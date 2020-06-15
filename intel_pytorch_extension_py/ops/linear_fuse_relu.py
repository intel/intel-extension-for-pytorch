import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
import math
import _torch_ipex as core

class LinearFuseReluFC(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = core.linear_fuse_relu(input, weight, bias)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, output = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if bias == None:
            output_mask = (input.requires_grad, weight.requires_grad, 0)
        else:
            output_mask = (input.requires_grad, weight.requires_grad, bias.requires_grad)
        grad_output = core.relu_use_dst_backward(grad_output, output)
        grad_input, grad_weight, grad_bias = core.linear_backward(input, grad_output, weight, output_mask)
        return (grad_input, grad_weight, grad_bias)

class LinearFuseRelu(nn.Module):
    r"""DNNL Linear module for using relu fused DNNL kernel"""

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(LinearFuseRelu, self).__init__()
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
            bound = 1 / math.sqrt(self.in_features)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # print(self.weight.shape)
        output = LinearFuseReluFC.apply(input, self.weight, self.bias)
        return output

