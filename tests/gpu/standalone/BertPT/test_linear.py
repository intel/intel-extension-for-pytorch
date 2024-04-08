import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_linear_1(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(1024, 1024, bias=True)
        x_cpu = torch.randn(
            (1, 1024),
            requires_grad=True,
            dtype=dtype,
        )

        z_cpu = linear(x_cpu)
        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        x_dpcpp = x_cpu.to("xpu")
        z_dpcpp = linear_dpcpp(x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)

    def test_linear_2(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(1024, 1024, bias=True)
        x_cpu = torch.randn(
            (1, 512, 1024),
            requires_grad=True,
            dtype=dtype,
        )

        z_cpu = linear(x_cpu)
        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        x_dpcpp = x_cpu.to("xpu")
        z_dpcpp = linear_dpcpp(x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)

    def test_linear_3(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(1024, 4096, bias=True)
        x_cpu = torch.randn(
            (1, 512, 1024),
            requires_grad=True,
            dtype=dtype,
        )

        z_cpu = linear(x_cpu)
        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        x_dpcpp = x_cpu.to("xpu")
        z_dpcpp = linear_dpcpp(x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)

    def test_linear_4(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(4096, 1024, bias=True)
        x_cpu = torch.randn(
            (1, 512, 4096),
            requires_grad=True,
            dtype=dtype,
        )

        z_cpu = linear(x_cpu)
        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        x_dpcpp = x_cpu.to("xpu")
        z_dpcpp = linear_dpcpp(x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)

    def test_linear_5(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(1024, 30522, bias=True)
        x_cpu = torch.randn(
            (1, 512, 1024),
            requires_grad=True,
            dtype=dtype,
        )

        z_cpu = linear(x_cpu)
        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        x_dpcpp = x_cpu.to("xpu")
        z_dpcpp = linear_dpcpp(x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)
