import torch
import torch.nn as nn
from typing import Optional


class BloomGELU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class GEGELU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def quick_gelu(self, x):
        return x * torch.sigmoid(1.702 * x)

    def forward(self, input, limit: Optional[float] = None):
        # input: [..., 2 * intermediate_size]
        a_gelu, a_linear = input[..., ::2], input[..., 1::2]
        if limit is not None:
            a_gelu = torch.where(
                torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
            )
            a_linear = torch.where(
                torch.isinf(a_linear), a_linear, a_linear.clamp(min=-limit, max=limit)
            )
        out_gelu = self.quick_gelu(a_gelu)
        return out_gelu * (a_linear + 1)


ACT2CLS = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "bloom_gelu": nn.GELU(approximate="tanh"),
    "gegelu": GEGELU(),
}

ACT2FN = ACT2CLS
