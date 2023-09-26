import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Union

from .Activation import ACT2FN
from .._transformer_configuration import IPEXTransformerConfig
import os
import math
import dataclasses
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import Tensor


class IPEXTransformerLinear(nn.Module):
    def __init__(self, weight=None, bias=None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias
    
    def forward(self, input):
        return torch.ops.torch_ipex.matmul_bias_out(input, self.weight, self.bias)


class IPEXTransformerQLinear(nn.Module):
    def __init__(self, weight=None, bias=None, scale=None, zp=None, gs=None):
        super(IPEXTransformerQLinear, self).__init__()
        # we set the weight and bias to None to avoid any possible memory presure
        self.weight = None
        self.bias = None
        self.scale = None
        self.zp = None
        self.gs = None

    def forward(self, input):
        return torch.ops.torch_ipex.mm_int4(input, self.weight, self.scale, self.zp, self.gs)


def matmul_add_add(attn_output, weight, tp_size=1, bias=None, residual=None):
    seq_len, bs, _ = attn_output.size()
    if residual is None:
        attn_output = torch.matmul(attn_output, weight)
        if bias is not None:
            attn_output += bias
    else:
        if bias is not None:
            attn_output = torch.ops.torch_ipex.mm_bias_resadd(attn_output, weight, bias, 1.0/tp_size, residual, 1.0/tp_size)
        else:
            attn_output = torch.addmm(residual.flatten(0, -2), attn_output.flatten(0, -2), weight, beta=1.0/tp_size)
    attn_output = attn_output.view(seq_len, bs, -1)
    return attn_output

class INT4Linear(nn.Module):
    __constants__ = ['in_features', 'out_features', 'group_size']
    in_features: int
    out_features: int
    group_size: int
    weight: Tensor
    scales: Tensor
    qzeros: Tensor

    def __init__(self, in_features: int, out_features: int, group_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        q_out_features = math.ceil(out_features / 8 * 4)
        self.in_features = in_features
        self.out_features = q_out_features
        self.group_size = in_features if group_size == -1 else group_size 

        self.qweight = Parameter(torch.empty((in_features, q_out_features),**factory_kwargs))
        self.scales = Parameter(torch.empty((math.ceil(in_features / self.group_size), out_features), **factory_kwargs))
        self.qzeros = Parameter(torch.empty((math.ceil(in_features / self.group_size), q_out_features), **factory_kwargs))
        #self.group_size = Parameter(group_size)

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.qweight.data = self.qweight.requires_grad_(False).byte()
        self.qzeros.data = self.qzeros.requires_grad_(False).byte()

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] == 1:
            if self.bias is not None:
                return torch.ops.torch_ipex.mm_int4(input, self.qweight, self.scales, self.qzeros, self.group_size)
            else:
                return torch.ops.torch_ipex.mm_bias_int4(input, self.qweight, self.bias, self.scales, self.qzeros, self.group_size)
        else:
            return torch.nn.functional.linear(input, self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}, bias={self.bias is not None}'
