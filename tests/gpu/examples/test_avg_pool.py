import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_avg_pool2d(self, dtype=torch.float):
        x_cpu = torch.ones([2, 2, 3, 3], device=cpu_device)
        grad_cpu = torch.ones([2, 2, 3, 3], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        x_cpu.requires_grad_(True)
        # y_cpu = conv1(x_cpu)
        y_cpu = avg_pool(x_cpu)
        print("y_cpu", y_cpu)
        # conv1.zero_grad()
        output_cpu = y_cpu.backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        # conv1.to("xpu")
        avg_pool.to(dpcpp_device)
        x_dpcpp.requires_grad_(True)
        # y_dpcpp = conv1(x_dpcpp)
        y_dpcpp = avg_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp.to("cpu"))
        # conv1.zero_grad()
        output_dpcpp = y_dpcpp.backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.to("cpu"))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x = torch.randn(1, 2, 3, 3, dtype=torch.float)
        w = torch.randn(2, 2, 3, 3, dtype=torch.float)
        conv = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        relu = torch.nn.ReLU()
        conv.weight.data = w
        ref = conv(x)
        ref = relu(ref)
        ref = avg_pool(ref)

        x = x.to("xpu").to(memory_format=torch.channels_last)
        w = w.to("xpu")
        conv.weight.data = w
        real = conv(x)
        real = relu(real)
        real = avg_pool(real)
        real = real.contiguous().cpu()

        print(real)
        print(ref)

        self.assertEqual(real, ref)
