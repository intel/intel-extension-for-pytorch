import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_hardshrink(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(8, 8)
        tanh = nn.Hardshrink()

        print("linear weight", linear.weight)

        x_cpu = torch.ones([1, 1, 8, 8], device=cpu_device, dtype=dtype)
        print("x_cpu", x_cpu)

        z_cpu = linear(x_cpu)
        print("z_cpu", z_cpu)

        y_cpu = tanh(z_cpu)
        print("y_cpu", y_cpu)

        y_cpu.backward(torch.ones([1, 1, 8, 8], device=cpu_device))
        print("linear grad", linear.weight.grad)

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        tanh_dpcpp = tanh.to("xpu")

        print("dpcpp linear weight", linear_dpcpp.weight.cpu())

        x_dpcpp = x_cpu.to("xpu")
        print("x_dpcpp", x_dpcpp.cpu())

        z_dpcpp = linear_dpcpp(x_dpcpp)
        print("z_dpcpp", z_dpcpp.cpu())

        y_dpcpp = tanh(z_dpcpp)
        print("y_dpcpp", y_dpcpp.cpu())

        y_dpcpp.backward(torch.ones([1, 1, 8, 8], device=dpcpp_device))
        print("dpcpp linear grad", linear.weight.grad.cpu())
        self.assertEqual(linear.weight.grad,
                         linear_dpcpp.weight.grad.to(cpu_device))
