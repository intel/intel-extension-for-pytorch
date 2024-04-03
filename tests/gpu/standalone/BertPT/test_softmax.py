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
        x_cpu = torch.randn((1, 16, 512, 512) , device=cpu_device)
        y_cpu_output = torch.randn(x_cpu.shape)
        x_dpcpp = x_cpu.clone().to("xpu")
        y_dpcpp_output = y_cpu_output.clone().to("xpu")
        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        y = F.log_softmax(x_cpu, 1)
        y.backward(y_cpu_output)
        y_dpcpp = F.log_softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

