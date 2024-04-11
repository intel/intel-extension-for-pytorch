from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_softmax(self, dtype=torch.float):
        x_cpu = torch.randn((1, 16, 512, 512), device=cpu_device, dtype=dtype)
        y_cpu_output = torch.randn(x_cpu.shape, dtype=dtype)
        x_dpcpp = x_cpu.clone().to("xpu")
        y_dpcpp_output = y_cpu_output.clone().to("xpu")
        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        y = F.log_softmax(x_cpu, 1)
        y.backward(y_cpu_output)
        y_dpcpp = F.log_softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_softmax_bfloat16(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 16, 512, 512), device=cpu_device, dtype=dtype)
        y_cpu_output = torch.randn(x_cpu.shape, dtype=dtype)
        x_dpcpp = x_cpu.clone().to("xpu")
        y_dpcpp_output = y_cpu_output.clone().to("xpu")
        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        y = F.log_softmax(x_cpu, 1)
        y.backward(y_cpu_output)
        y_dpcpp = F.log_softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_softmax_float16(self, dtype=torch.float):
        x_cpu = torch.randn((1, 16, 512, 512), device=cpu_device, dtype=dtype)
        y_cpu_output = torch.randn(x_cpu.shape, dtype=dtype)
        dtype_dpcpp = torch.float16
        x_dpcpp = x_cpu.clone().to("xpu").to(dtype_dpcpp)
        y_dpcpp_output = y_cpu_output.clone().to("xpu").to(dtype_dpcpp)
        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        y = F.log_softmax(x_cpu, 1)
        y.backward(y_cpu_output)
        y_dpcpp = F.log_softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu().to(torch.float), rtol=1e-2, atol=1e-2)

