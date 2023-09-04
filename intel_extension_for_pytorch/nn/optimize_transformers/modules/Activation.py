import torch
import torch.nn as nn

class BloomGELU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

ACT2CLS = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate='tanh'),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "bloom_gelu": nn.GELU(approximate='tanh') 
}

ACT2FN = ACT2CLS
