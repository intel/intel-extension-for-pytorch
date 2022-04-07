import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_linear(self, dtype=torch.float):
        # cpu
        linear = nn.Linear(4, 2)
        tanh = nn.Tanh()

        linear_weight_cpu = linear.weight.clone()
        print("linear weight", linear_weight_cpu)

        x_cpu = torch.tensor([[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]],
                             requires_grad=True, device=cpu_device, dtype=dtype)
        print("x_cpu", x_cpu)

        z_cpu = linear(x_cpu)
        print("z_cpu", z_cpu)

        y_cpu = tanh(z_cpu)
        print("y_cpu", y_cpu)

        y_cpu.backward(torch.tensor([[1.01, 8.32], [2.4, 3.22]], device=cpu_device))
        x_grad_cpu = x_cpu.grad.clone()
        linear_weight_grad_cpu = linear.weight.grad.clone()
        print("cpu input grad", x_grad_cpu)
        print("cpu linear grad", linear_weight_grad_cpu)

        linear.zero_grad()

        # dpcpp
        linear_dpcpp = linear.to("xpu")
        tanh_dpcpp = tanh.to("xpu")

        linear_weight_dpcpp = linear_dpcpp.weight.clone()
        print("dpcpp linear weight", linear_weight_dpcpp.cpu())

        x_dpcpp = torch.tensor([[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]],
                               requires_grad=True, device=dpcpp_device, dtype=dtype)
        print("x_dpcpp", x_dpcpp.to("cpu"))

        z_dpcpp = linear_dpcpp(x_dpcpp)
        print("z_dpcpp", z_dpcpp.to("cpu"))

        y_dpcpp = tanh(z_dpcpp)
        print("y_dpcpp", y_dpcpp.to("cpu"))

        y_dpcpp.backward(torch.tensor([[1.01, 8.32], [2.4, 3.22]], device=dpcpp_device))
        x_grad_dpcpp = x_dpcpp.grad.clone()
        linear_weight_grad_dpcpp = linear_dpcpp.weight.grad.clone()
        print("dpcpp input grad", x_grad_dpcpp.cpu())
        print("dpcpp linear grad", linear_weight_grad_dpcpp.cpu())

        self.assertEqual(x_cpu, x_dpcpp)
        self.assertEqual(z_cpu, z_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)
        self.assertEqual(x_grad_cpu, x_grad_dpcpp)
        self.assertEqual(linear_weight_cpu, linear_weight_dpcpp)
        self.assertEqual(linear_weight_grad_cpu, linear_weight_grad_dpcpp)

        # new added case for the shared weights in one tensor
        # functionality
        x_cpu = torch.ones([3, 4], device=cpu_device, dtype=dtype)
        grad_cpu = torch.ones([3, 2], device=cpu_device, dtype=dtype)
        weight = torch.ones([3, 8], device=cpu_device, dtype=dtype)

        weight[:, 4:] = 2

        print(x_cpu)
        print(weight)

        print(weight[:, :4])
        y1_cpu = F.linear(x_cpu, weight[:, :4])
        # print(y_cpu)
        y2_cpu = F.linear(x_cpu, weight[:, 4:])
        # print(y_cpu)

        print("--------------------------------------------------------------------")

        x_sycl = x_cpu.to(dpcpp_device)
        weight_sycl = weight.to(dpcpp_device)
        # print(x_sycl.cpu())
        # print(weight_sycl.cpu())

        y1_sycl = F.linear(x_sycl, weight_sycl[:, :4])
        # print(y_sycl.cpu())
        y2_sycl = F.linear(x_sycl, weight_sycl[:, 4:])
        # print(y_sycl.cpu())
        self.assertEqual(x_cpu, x_sycl)
        self.assertEqual(weight, weight_sycl)
        self.assertEqual(y1_cpu, y1_sycl)
        self.assertEqual(y2_cpu, y2_sycl)
