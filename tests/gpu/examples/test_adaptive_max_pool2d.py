import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_adaptive_max_pool2d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 1, 8, 8], device=cpu_device)
        grad_cpu = torch.randn([1, 1, 2, 2], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.cpu())

        self.assertEqual(y_cpu[0], y_dpcpp[0].to(dpcpp_device))
        # For now our MaxPooling return indices are wrong, oneDNN is trying to fix them
        # JIRA: https://jira.devtools.intel.com/browse/MFDNN-3672
        # self.assertEqual( y_cpu[1], y_dpcpp[1].to(dpcpp_device))
        self.assertEqual(x_dpcpp.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.randn([3, 2, 8, 8], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last)
        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)

        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(dpcpp_device))

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([3, 2, 8, 8], device=cpu_device)
        grad_cpu = torch.randn([3, 2, 2, 2], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.cpu())

        self.assertEqual(y_cpu[0], y_dpcpp[0].to(dpcpp_device))
        # For now our MaxPooling return indices are wrong, oneDNN is trying to fix them
        # JIRA: https://jira.devtools.intel.com/browse/MFDNN-3672
        # self.assertEqual( y_cpu[1], y_dpcpp[1].to(dpcpp_device))
        self.assertEqual(x_dpcpp.grad, x_dpcpp.grad.to(cpu_device))
