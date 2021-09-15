import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

import numpy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_pool3d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.randn([1, 4, 4, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        grad_dpcpp = grad_cpu.to("xpu")

        max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 3, 4], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last)

        max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])

        max_pool.to("xpu")
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.randn([1, 4, 4, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last)
        grad_dpcpp = grad_cpu.to("xpu")

        max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
