import torch
from torch.autograd import Function
import torch.nn.functional as F
import _torch_ipex as core
from typing import Optional

def linear(input, weight, bias: Optional[torch.Tensor] = None):
    if bias is None:
        bias = torch.zeros(weight.size(0))
    return torch.ops.torch_ipex.linear(input, weight, bias)

F.linear = linear
