import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_fractional_max_pool2d_bf16(self, dtype=torch.bfloat16):
        ctype = torch.float
        x_cpu = torch.randn([4, 4, 16, 16], device=cpu_device, dtype=ctype)
        x_dpcpp = x_cpu.to(dtype).to("xpu")
        grad_cpu = torch.randn([4, 4, 2, 2], device=cpu_device, dtype=ctype)
        grad_dpcpp = grad_cpu.to(dtype).to("xpu")
        max_pool = nn.FractionalMaxPool2d(
            2, output_size=(2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        y_cpu[0].backward(grad_cpu)

        max_pool = nn.FractionalMaxPool2d(
            2, output_size=(2, 2), return_indices=True)
        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        self.assertEqual(y_dpcpp[0].is_contiguous(), True)
        self.assertEqual(x_dpcpp.grad.is_contiguous(), True)
        self.assertEqual(y_cpu[0], y_dpcpp[0].float().to(cpu_device), rtol=10e-4, atol=10e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.float().to(cpu_device), rtol=10e-4, atol=10e-2)

    def test_fractional_max_pool3d_bf16(self, dtype=torch.bfloat16):
        ctype = torch.float
        x_cpu = torch.randn([4, 4, 16, 16, 16], device=cpu_device, dtype=ctype)
        x_dpcpp = x_cpu.to(dtype).to("xpu")
        grad_cpu = torch.randn([4, 4, 2, 2, 2], device=cpu_device, dtype=ctype)
        grad_dpcpp = grad_cpu.to(dtype).to("xpu")
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
        self.assertEqual(y_cpu[0], y_dpcpp[0].float().to(cpu_device), rtol=10e-4, atol=10e-2)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.float().to(cpu_device), rtol=10e-4, atol=10e-2)
