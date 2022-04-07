import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_hardtanh(self, dtype=torch.float):
        grad_output_cpu = torch.randn([1, 1, 8, 8])
        grad_output_dpcpp = grad_output_cpu.xpu()

        # cpu
        linear = nn.Linear(8, 8)
        tanh = nn.Hardtanh()
        print("linear weight", linear.weight)
        x_cpu = torch.ones([1, 1, 8, 8], device=cpu_device, dtype=dtype)
        print("x_cpu", x_cpu)
        z_cpu = linear(x_cpu)
        print("z_cpu", z_cpu)
        y_cpu = tanh(z_cpu)
        print("y_cpu", y_cpu)
        y_cpu.backward(grad_output_cpu)
        linear_weight_grad_cpu = linear.weight.grad.clone()
        print("linear grad", linear_weight_grad_cpu)
        linear.zero_grad()

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
        y_dpcpp.backward(grad_output_dpcpp)
        linear_weight_grad_dpcpp = linear.weight.grad.clone()
        print("dpcpp linear grad", linear_weight_grad_dpcpp)
        linear_dpcpp.zero_grad()

        self.assertEqual(z_cpu, z_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)
        self.assertEqual(linear_weight_grad_cpu, linear_weight_grad_dpcpp)
