import torch
import torch.nn.functional
from torch.testing._internal.common_utils import TestCase

import ipex
import copy
import pytest


class TestNNMethod(TestCase):

    def test_activation_relu(self, dtype=torch.float):
        relu_ = torch.nn.functional.relu_
        relu = torch.nn.functional.relu
        x_cpu = torch.tensor([[-0.1, 0.2], [-0.2, 0.3], [0.4, 0.5], [0.5, -0.6]])
        x_dpcpp = x_cpu.to("xpu")

        relu_(x_cpu)
        relu_(x_dpcpp)
        print("cpu relu_ ", x_cpu)
        print("dpcpp relu_ ", x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())

        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = relu(x_cpu)
        y_dpcpp = relu(x_dpcpp)
        print("cpu relu ", y_cpu)
        print("dpcpp relu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu relu bwd", x_cpu.grad)
        print("dpcpp relu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_relu_channels_last(self, dtype=torch.float):
        x = torch.randn(1, 2, 3, 3, dtype=torch.float)
        w = torch.randn(2, 2, 3, 3, dtype=torch.float)
        conv = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
        bn = torch.nn.BatchNorm2d(2)
        relu = torch.nn.ReLU()
        conv.weight.data = w
        ref = conv(x)
        ref = bn(ref)
        ref = relu(ref)

        x = x.to("xpu").to(memory_format=torch.channels_last)
        w = w.to("xpu").to(memory_format=torch.channels_last)
        bn = bn.to("xpu")
        conv.weight.data = w
        real = conv(x)
        real = bn(real)
        real = relu(real)
        real = real.contiguous().cpu()

        print(real)
        print(ref)
        self.assertEqual(real, ref)

    def test_activation_relu_channels_last_bwd(self, dtype=torch.float):
        relu = torch.nn.functional.relu
        x_cpu = torch.randn(1, 2, 3, 3, dtype=torch.float)
        grad_cpu = torch.randn(1, 2, 3, 3, dtype=torch.float)
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last)
        grad_dpcpp = grad_cpu.to("xpu")

        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = relu(x_cpu)
        y_dpcpp = relu(x_dpcpp)
        print("cpu relu ", y_cpu)
        print("dpcpp relu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(grad_cpu)
        y_dpcpp.backward(grad_dpcpp)

        print("cpu relu bwd", x_cpu.grad)
        print("dpcpp relu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_rrelu(self, dtype=torch.float):
        #  Will not check the result due to different random seeds on cpu and xpu
        RReLU = torch.nn.RReLU(0.1, 0.3)
        RReLU_dpcpp = copy.deepcopy(RReLU).to("xpu")
        x_cpu = torch.tensor([[-0.1, 0.2], [-0.2, 0.3], [0.4, 0.5], [0.5, -0.6]])
        x_dpcpp = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = RReLU(x_cpu)
        y_dpcpp = RReLU_dpcpp(x_dpcpp)
        print("cpu rrelu ", y_cpu)
        print("dpcpp rrelu ", y_dpcpp.cpu())
        #  self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu rrelu bwd", x_cpu.grad)
        print("dpcpp rrelu bwd", x_dpcpp.grad.cpu())
        #  self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_gelu(self, dtype=torch.float):
        GELU = torch.nn.GELU()
        GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
        x_cpu = torch.tensor([[-0.1, 0.2], [-0.2, 0.3], [0.4, 0.5], [0.5, -0.6]])
        x_dpcpp = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = GELU(x_cpu)
        y_dpcpp = GELU_dpcpp(x_dpcpp)
        print("cpu gelu ", y_cpu)
        print("dpcpp gelu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        # y_cpu = torch.tensor([[1, 1],[1, 1],[1, 1],[1, 1]]);
        # y_dpcpp = y_cpu.to("xpu")
        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu gelu bwd", x_cpu.grad)
        print("dpcpp gelu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_activation_prelu(self, dtype=torch.float):
        PReLU = torch.nn.PReLU()
        PReLU_dpcpp = copy.deepcopy(PReLU).to("xpu")
        x_cpu = torch.tensor([[-0.1, 0.2], [-0.2, 0.3], [0.4, 0.5], [0.5, -0.6]])
        x_dpcpp = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = PReLU(x_cpu)
        y_dpcpp = PReLU_dpcpp(x_dpcpp)
        print("cpu prelu ", y_cpu)
        print("dpcpp prelu ", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        print("cpu prelu bwd", x_cpu.grad)
        print("dpcpp prelu bwd", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
