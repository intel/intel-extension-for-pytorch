import os
import warnings
from typing import Union, Optional, Callable, Tuple, Dict, List, Any
from functools import partial
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import Tensor

class INT4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.qweight = Parameter(None)
        self.weight = Parameter(None)
        self.scales = Parameter(None)
        self.qzeros = Parameter(None)
        self.group_size = Parameter(None)
        if bias:
            self.bias = Parameter(None)
        else:
            self.register_parameter('bias', None)
        self.qweight.data = self.qweight.requires_grad_(False).byte()
        self.qzeros.data = self.qzeros.requires_grad_(False).byte()
        self.group_size.data = self.group_size.requires_grad_(False).int()

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] == 1:
            if self.bias is not none:
                return torch.ops.torch_ipex.mm_int4(input, self.qweight, self.scales, self.qzeros, self.group_size)
            else:
                return torch.ops.torch_ipex.mm_bias_int4(input, self.qweight, self.bias, self.scales, self.qzeros, self.group_size)
        else:
            return torch.nn.functional.linear(input, self.weight, bias=self.bias)
