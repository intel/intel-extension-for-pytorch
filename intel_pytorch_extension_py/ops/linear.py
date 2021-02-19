import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core
from typing import Optional

def linear(input, weight, bias: Optional[torch.Tensor] = None):
    return torch.ops.torch_ipex.linear(input, weight, bias)

F.linear = linear

class LinearRelu(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearRelu, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        return torch.ops.torch_ipex.linear_relu(input, self.weight, self.bias)