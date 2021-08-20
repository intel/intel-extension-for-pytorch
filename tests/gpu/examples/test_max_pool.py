import numpy
import torch
import torch.nn as nn
import ipex
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_pool(self, dtype=torch.float):
        x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        grad_dpcpp = grad_cpu.to("xpu")

        conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        max_pool = nn.MaxPool2d(kernel_size=3, stride=1,
                                padding=1, return_indices=True)

        x_cpu.requires_grad_(True)
        y_cpu1 = conv1(x_cpu)
        y_cpu = max_pool(y_cpu1)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        conv1.to("xpu")
        max_pool.to("xpu")
        
        x_dpcpp.requires_grad_(True)
        y_dpcpp1 = conv1(x_dpcpp)
        y_dpcpp = max_pool(y_dpcpp1)
        print("y_dpcpp", y_dpcpp[0].to("cpu"))
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x = torch.randn(1, 2, 3, 3, dtype=torch.float)
        w = torch.randn(2, 2, 3, 3, dtype=torch.float)
        conv = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        relu = torch.nn.ReLU()
        conv.weight.data = w
        ref = conv(x)
        ref = relu(ref)
        ref = max_pool(ref)
        ref = ref[0]

        x = x.to("xpu").to(memory_format=torch.channels_last)
        w = w.to("xpu")
        conv.weight.data = w
        real = conv(x)
        real = relu(real)
        real = max_pool(real)
        real = real[0].contiguous().cpu()

        print(real)
        print(ref)

        self.assertEqual(real, ref)

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
        grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last)
        grad_dpcpp = grad_cpu.to("xpu")

        conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
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