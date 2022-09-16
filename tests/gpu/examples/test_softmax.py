from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa


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
        x_cpu = torch.tensor([[[1.5357e+00, -2.4013e+01, -9.2085e+01],
                               [6.2914e-01, 6.7819e+01, -9.3087e+01],
                               [2.2412e+00, -1.0471e+02, -1.3249e+02]]])

        y_cpu_output = torch.randn(x_cpu.shape)
        x_dpcpp = x_cpu.clone().to("xpu")
        y_dpcpp_output = y_cpu_output.clone().to("xpu")

        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        self.softmax_basic(x_cpu, y_cpu_output, x_dpcpp, y_dpcpp_output)

        x_cpu = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]])
        y_cpu_output = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]])
        x_dpcpp = x_cpu.clone().to("xpu")
        y_dpcpp_output = y_cpu_output.clone().to("xpu")

        x_cpu.requires_grad_()
        x_dpcpp.requires_grad_()
        self.softmax_basic(x_cpu, y_cpu_output, x_dpcpp, y_dpcpp_output)

        shape = [[8], [7, 8], [7, 8, 512], [16, 7, 8, 512], [16, 7, 8, 512, 35]]
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                dim = j - 1
                x = torch.randn(shape[i])
                grad = torch.randn(shape[i])
                x_cpu = x.clone().requires_grad_()
                y_cpu = F.softmax(x_cpu, dim)
                y_cpu.backward(grad.clone())

                x_dpcpp = x.clone().to(dpcpp_device).requires_grad_()
                y_dpcpp = F.softmax(x_dpcpp, dim)
                y_dpcpp.backward(grad.clone().to(dpcpp_device))
                self.assertEqual(y_cpu, y_dpcpp.cpu())
                self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_softmax_non_contiguous(self, dtype=torch.float):

        x_cpu = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]],
                             requires_grad=True, device=cpu_device).as_strided([2, 3], [1, 2])
        y_cpu_output = torch.tensor(
            [[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]], requires_grad=True, device=cpu_device)

        x_dpcpp = torch.tensor([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]],
                               requires_grad=True, device=dpcpp_device).as_strided([2, 3], [1, 2])
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

        self.assertEqual(y, y_dpcpp.cpu())
        print("log_softmax dpcpp:", y_dpcpp.cpu())
        if x_dpcpp.grad is None:
            print("log_softmax x_dpcpp_grad = ", None)
            self.assertEqual(x_cpu.grad, None)
        else:
            print("log_softmax x_dpcpp_grad = ", x_dpcpp.grad.cpu())
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

        y = F.softmax(x_cpu, 1)
        y.backward(y_cpu_output)

        print("softmax:", y)
        print("softmax x_cpu_grad = ", x_cpu.grad)

        y_dpcpp = F.softmax(x_dpcpp, 1)
        y_dpcpp.backward(y_dpcpp_output)

        self.assertEqual(y, y_dpcpp.cpu())
        print("softmax dpcpp:", y_dpcpp.cpu())
        if x_dpcpp.grad is None:
            print("softmax x_dpcpp_grad = ", None)
            self.assertEqual(x_cpu.grad, None)
        else:
            print("softmax x_dpcpp_grad = ", x_dpcpp.grad.cpu())
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

    def test_softmax_bwd_non_contiguous(self, dtype=torch.float):
        x_cpu = torch.tensor([[[[0.5, 1.5, 0.1],
                                [2.2, 1.3, 1.7],
                                [0.1, 1.1, 0.8]]]],
                             requires_grad=True, device=cpu_device).as_strided([1, 1, 3, 3], [9, 9, 1, 3])
        y_cpu_output = torch.tensor([[[[0.5, 1.5, 0.1],
                                       [2.2, 1.3, 1.7],
                                       [0.1, 1.1, 0.8]]]],
                                    requires_grad=True, device=cpu_device)

        x_dpcpp = torch.tensor([[[[0.5, 1.5, 0.1],
                                [2.2, 1.3, 1.7],
                                [0.1, 1.1, 0.8]]]],
                               requires_grad=True, device=dpcpp_device).as_strided([1, 1, 3, 3], [9, 9, 1, 3])
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

        self.assertEqual(y, y_dpcpp.cpu())
        print("softmax dpcpp:", y_dpcpp.cpu())
        if x_dpcpp.grad is None:
            print("softmax x_dpcpp_grad = ", None)
            self.assertEqual(x_cpu.grad, None)
        else:
            print("softmax x_dpcpp_grad = ", x_dpcpp.grad.cpu())
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
