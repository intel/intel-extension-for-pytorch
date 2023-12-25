from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def softmax_basic(self, x_cpu, y_cpu_output, x_dpcpp, y_dpcpp_output):
        print("x:", x_cpu)
        print("x_dpcpp:", x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())

        y = F.log_softmax(x_cpu, 1)
        y.backward(y_cpu_output)

        print("log_softmax:", y)
        print("log_softmax x_cpu_grad = ", x_cpu.grad)

        y_dpcpp = F.log_softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)

        print("log_softmax dpcpp:", y_dpcpp.cpu())
        print("log_softmax x_dpcpp_grad = ", x_dpcpp.grad.cpu())
        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

        y = F.softmax(x_cpu, 1)
        y.backward(y_cpu_output)

        print("softmax:", y)
        print("softmax x_cpu_grad = ", x_cpu.grad)

        y_dpcpp = F.softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)

        print("softmax dpcpp:", y_dpcpp.cpu())
        print("softmax x_dpcpp_grad = ", x_dpcpp.grad.cpu())
        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_softmax(self, dtype=torch.float):
        x_cpu = torch.randn((2, 16, 384, 384) , device=cpu_device)
        y_cpu_output = torch.randn(x_cpu.shape)
        x_dpcpp = x_cpu.clone().to("xpu")
        y_dpcpp_output = y_cpu_output.clone().to("xpu")
        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        self.softmax_basic(x_cpu, y_cpu_output, x_dpcpp, y_dpcpp_output)
