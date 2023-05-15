import unittest
import torch
import torch.nn as nn
from common_utils import TestCase
import intel_extension_for_pytorch as ipex


class Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.conv(x)


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        return self.linear(x)


class MatmulDiv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.transpose(x, -1, -2).contiguous()
        z = torch.matmul(x, y)
        return z.div(2.0)


class Tester(TestCase):
    def test_a_lru_cache_resize(self):
        import os

        # Set LRU_CACHE_CAPACITY < 1024 to trigger resize
        os.environ["LRU_CACHE_CAPACITY"] = "512"
        # Conv
        conv = Conv2d().eval()
        conv = ipex.optimize(conv, dtype=torch.float32)
        conv(torch.randn(3, 64, 56, 56))
        # Linear
        linear = Linear().eval()
        linear = ipex.optimize(linear, dtype=torch.bfloat16)
        linear(torch.randn((100, 64), dtype=torch.bfloat16))
        # Matmul
        matmul = MatmulDiv().eval()
        x = torch.randn(10, 3, 4)
        traced_model = torch.jit.trace(matmul, x).eval()
        traced_model.graph_for(x)
        # unset this environment variable
        del os.environ["LRU_CACHE_CAPACITY"]


if __name__ == "__main__":
    test = unittest.main()
