import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_fractional_max_pool2d(self, dtype=torch.float):
        x_cpu = torch.randn([2, 2, 4, 4], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_cpu = torch.randn([2, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        max_pool = nn.FractionalMaxPool2d(
            2, output_size=(2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        y_cpu[0].backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        max_pool = nn.FractionalMaxPool2d(
            2, output_size=(2, 2), return_indices=True)
        max_pool.to(dpcpp_device)
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        print("y_dpcpp", y_dpcpp[0].cpu())
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        y_dpcpp[0].backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(y_dpcpp[0].is_contiguous(), True)
        self.assertEqual(x_dpcpp.grad.is_contiguous(), True)
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    def test_fractional_max_pool2d_channels_last(self, dtype=torch.float):
        x_cpu = torch.randn([2, 2, 4, 4], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(memory_format=torch.channels_last).to(dpcpp_device)
        grad_cpu = torch.randn([2, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        max_pool = nn.FractionalMaxPool2d(
            2, output_size=(2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("\ny_cpu:\n", y_cpu[0])
        y_cpu[0].backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        max_pool = nn.FractionalMaxPool2d(
            2, output_size=(2, 2), return_indices=True)
        max_pool.to(dpcpp_device)
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        print("\ny_dpcpp:\n", y_dpcpp[0].cpu())
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        y_dpcpp[0].backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(y_dpcpp[0].is_contiguous(memory_format=torch.channels_last), True)
        self.assertEqual(x_dpcpp.grad.is_contiguous(memory_format=torch.channels_last), True)
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))
