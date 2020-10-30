import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import math
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core
from typing import Optional

def linear(input, weight, bias: Optional[torch.Tensor] = None):
    return torch.ops.torch_ipex.linear(input, weight, bias)

F.linear = linear


class LinearRelu(nn.Module):
    r"""DNNL Linear module for using relu fused DNNL kernel"""

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True, fuse_relu=False):
        super(LinearRelu, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.fuse_relu = fuse_relu

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
    
    def prepack_weight(self):
        prepacked_weight = Parameter(core.linear_prepack_weight(self.weight))
        self.weight = prepacked_weight

    def forward(self, input):
        ret = torch.ops.torch_ipex.linear_relu(input, self.weight, self.bias, self.fuse_relu)
        return ret