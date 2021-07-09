from __future__ import print_function
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_softmax(self, dtype=torch.float):

        x_cpu = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]],
                             requires_grad=True, device=cpu_device)
        y_cpu_output = torch.tensor(
            [[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device=cpu_device)

        x_dpcpp = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]],
                               requires_grad=True, device=dpcpp_device)
        y_dpcpp_output = torch.tensor(
            [[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device=dpcpp_device)

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

    def test_softmax_bwd(self, dtype=torch.float):
        x_cpu = torch.tensor([[[[0.5, 1.5, 0.1],
                                [2.2, 1.3, 1.7],
                                [0.1, 1.1, 0.8]]]],
                             requires_grad=True, device=cpu_device)
        y_cpu_output = torch.tensor([[[[0.5, 1.5, 0.1],
                                [2.2, 1.3, 1.7],
                                [0.1, 1.1, 0.8]]]],
                             requires_grad=True, device=cpu_device)

        x_dpcpp = torch.tensor([[[[0.5, 1.5, 0.1],
                                [2.2, 1.3, 1.7],
                                [0.1, 1.1, 0.8]]]],
                             requires_grad=True, device=dpcpp_device)
        y_dpcpp_output = torch.tensor([[[[0.5, 1.5, 0.1],
                                [2.2, 1.3, 1.7],
                                [0.1, 1.1, 0.8]]]],
                             requires_grad=True, device=dpcpp_device)

        print("x:", x_cpu)
        print("x_dpcpp:", x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())

        conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        y1 = conv(x_cpu)
        y = F.softmax(y1, 2)
        y.backward(y_cpu_output)

        print("softmax:", y)
        print("softmax x_cpu_grad = ", x_cpu.grad)

        conv.to(dpcpp_device)
        y_dpcpp1 = conv(x_dpcpp)
        y_dpcpp = F.softmax(y_dpcpp1, 2)
        y_dpcpp.backward(y_dpcpp_output)

        print("softmax dpcpp:", y_dpcpp.cpu())
        print("softmax x_dpcpp_grad = ", x_dpcpp.grad.cpu())
        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())