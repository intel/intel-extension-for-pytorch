import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest


class TestTorchMethod(TestCase):
    def test_aminmax(self, dtype=torch.float):
        src_cpu = torch.randn(2, 5)
        dst_cpu = torch._aminmax(src_cpu)
        # print("input = ", src_cpu)
        # print("cpu result:", dst_cpu)

        src_gpu = src_cpu.to("xpu")
        dst_gpu = torch._aminmax(src_gpu)
        # print("gpu result:", dst_gpu[0].cpu(), dst_gpu[1].cpu())

        self.assertEqual(dst_cpu, dst_gpu)
