import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")

class TestTensorMethod(TestCase):
    def test_permute(self, dtype=torch.float):
        src = torch.randn((1, 2, 2), device=cpu_device)
        print("raw src strides: ", src.stride())

        raw = src.permute(2, 0, 1).contiguous()
        print("raw dst strides: ", raw.stride())
        print("raw: ", raw)

        src_dpcpp = src.to(dpcpp_device)
        print("real src strides: ", src.stride())
        real = src_dpcpp.permute(2, 0, 1).contiguous()
        print("raw dst strides: ", real.stride())
        print("real: ", real.cpu())

        self.assertEqual(raw, real.to(cpu_device))

    def test_simple_copy(self, dtype=torch.float):
        src = torch.randn((1, 2, 2), device=cpu_device)
        dst = torch.randn((1, 2, 2), device=cpu_device)

        dst.copy_(src)
        print("raw: ", dst)

        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        print("real: ", dst_dpcpp.cpu())

        self.assertEqual(dst, dst_dpcpp.to(cpu_device))
