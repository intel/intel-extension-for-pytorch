import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_fractional_max_pool3d(self, dtype=torch.float):
        x_cpu = torch.randn([4, 4, 16, 16, 16], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        grad_cpu = torch.randn([4, 4, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to("xpu")
        max_pool = nn.FractionalMaxPool3d(
            2, output_size=(2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        y_cpu[0].backward(grad_cpu)

        max_pool = nn.FractionalMaxPool3d(
            2, output_size=(2, 2, 2), return_indices=True)
        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        self.assertEqual(y_dpcpp[0].is_contiguous(), True)
        self.assertEqual(x_dpcpp.grad.is_contiguous(), True)
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    @pytest.mark.skipif(torch.xpu.using_onednn_layout(), reason="channels last does not support onednn block format")
    def test_fractional_max_pool3d_channels_last(self, dtype=torch.float):
        x_cpu = torch.randn([4, 4, 16, 16, 16], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(memory_format=torch.channels_last_3d).to("xpu")
        max_pool = nn.FractionalMaxPool3d(
            2, output_size=(2, 2, 2), return_indices=True)
        grad_cpu = torch.randn([4, 4, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to("xpu")

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        y_cpu[0].backward(grad_cpu)

        max_pool = nn.FractionalMaxPool3d(
            2, output_size=(2, 2, 2), return_indices=True)
        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        self.assertEqual(y_dpcpp[0].is_contiguous(memory_format=torch.channels_last_3d), True)
        self.assertEqual(x_dpcpp.grad.is_contiguous(memory_format=torch.channels_last_3d), True)
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))
