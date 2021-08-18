import torch
from torch import nn

def frozen_batch_norm(x, weight, bias, running_mean, running_var):
    return torch.ops.torch_ipex.frozen_batch_norm(x, weight, bias, running_mean, running_var)

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        return frozen_batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var)
