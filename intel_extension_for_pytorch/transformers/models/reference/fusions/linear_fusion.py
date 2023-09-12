import torch
from torch import nn
import math


class _IPEXlinearSiluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x):
        return nn.functional.silu(self.linear(x))


class _IPEXlinearAddRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x, y):
        return self.linear(x) + y


class _IPEXlinearAddAddRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x, y, z):
        return self.linear(x) + y + z


class _IPEXlinearMulRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x, y):
        return self.linear(x) * y


class _IPEXlinearNewGeluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x):
        x = self.linear(x)
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class _IPEXlinearReluRef(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.linear = module

    def forward(self, x):
        return nn.functional.relu(self.linear(x))
