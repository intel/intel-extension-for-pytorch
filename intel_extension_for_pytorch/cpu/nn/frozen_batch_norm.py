import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    r"""
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): ``C`` from an expected input of size ``(N, C, H, W)``.
            Input shape: ``(N, C, H, W)``.
            Output shape: ``(N, C, H, W)`` (same shape as input).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, input):
        return torch.ops.torch_ipex.frozen_batch_norm(
            input, self.weight, self.bias, self.running_mean, self.running_var, self.eps
        )
