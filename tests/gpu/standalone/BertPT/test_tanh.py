import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest

shapes = [
        (1, 1024)
        ]

class TestNNMethod(TestCase):
    def test_tanh(self, dtype=torch.float):
        for shape in shapes:
            # cpu
            tanh = nn.Tanh()

            x_cpu = torch.randn(shape, dtype=dtype)
            x_cpu.requires_grad_(True)
            z_cpu = tanh(x_cpu)
            z_cpu.backward(torch.ones_like(x_cpu))

            # dpcpp
            x_dpcpp = x_cpu.to("xpu")
            z_dpcpp = tanh(x_dpcpp)
            z_dpcpp.backward(torch.ones_like(x_dpcpp))

            self.assertEqual(x_cpu, x_dpcpp.cpu())
            self.assertEqual(z_cpu, z_dpcpp.cpu())

    def test_tanh_bfloat16(self, dtype=torch.bfloat16):
        for shape in shapes:
            # cpu
            tanh = nn.Tanh()

            x_cpu = torch.randn(shape, dtype=dtype)
            x_cpu.requires_grad_(True)
            z_cpu = tanh(x_cpu)
            z_cpu.backward(torch.ones_like(x_cpu))

            # dpcpp
            x_dpcpp = x_cpu.to("xpu")
            z_dpcpp = tanh(x_dpcpp)
            z_dpcpp.backward(torch.ones_like(x_dpcpp))

            self.assertEqual(x_cpu, x_dpcpp.cpu())
            self.assertEqual(z_cpu, z_dpcpp.cpu())

    def test_tanh_float16(self, dtype=torch.float16):
        for shape in shapes:
            # cpu
            tanh = nn.Tanh()

            x_cpu = torch.randn(shape, dtype=dtype)
            x_cpu.requires_grad_(True)
            z_cpu = tanh(x_cpu)
            z_cpu.backward(torch.ones_like(x_cpu))

            # dpcpp
            x_dpcpp = x_cpu.to("xpu")
            z_dpcpp = tanh(x_dpcpp)
            z_dpcpp.backward(torch.ones_like(x_dpcpp))

            self.assertEqual(x_cpu, x_dpcpp.cpu())
            self.assertEqual(z_cpu, z_dpcpp.cpu())
