import torch
from torch.autograd import Function
import _torch_ipex as core

torch_reshape = torch.reshape

class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx, input, size):
        output = core.reshape(input, size)
        return output

def reshape(input, size):
    if input.device.type == 'dpcpp':
        return ReshapeFunction.apply(input, size)
    return torch_reshape(input, size)

torch.reshape = reshape