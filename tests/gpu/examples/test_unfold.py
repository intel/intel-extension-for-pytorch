import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_unflod(self, dtype=torch.float):
        x_cpu = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
        y = x_cpu.unfold(0, 2, 1)
        x_dpcpp = torch.tensor([1., 2., 3., 4., 5., 6., 7.],
                               device=torch.device("xpu"))
        y_dpcpp = x_dpcpp.unfold(0, 2, 1)
        print("unfold cpu ", y)
        print("unfold dpcpp ", y_dpcpp.to("cpu"))
        self.assertEqual(y, y_dpcpp.cpu())

    def test_unfold_backward(self, dtype=torch.float):
        x_cpu = torch.tensor([1., 2., 3., 4., 5., 6., 7.], requires_grad=True)
        x_xpu = torch.tensor([1., 2., 3., 4., 5., 6., 7.], requires_grad=True, device=dpcpp_device)  # a leaf node
        linear = torch.nn.Linear(5, 10)
        activation = torch.nn.ReLU()
        softmax = torch.nn.Softmax(dim=0)

        y_cpu = x_cpu.unfold(0, 5, 1)
        y_cpu = linear(y_cpu)
        y_cpu = activation(y_cpu)
        y_cpu = softmax(y_cpu)
        y_cpu = torch.mul(y_cpu, y_cpu)
        y_cpu = torch.sum(y_cpu)
        y_cpu.backward()

        y_xpu = x_xpu.unfold(0, 5, 1)
        y_xpu = linear.to(dpcpp_device)(y_xpu)
        y_xpu = activation.to(dpcpp_device)(y_xpu)
        y_xpu = softmax.to(dpcpp_device)(y_xpu)
        y_xpu = torch.mul(y_xpu, y_xpu)
        y_xpu = torch.sum(y_xpu)
        y_xpu.backward()

        self.assertEqual(x_cpu.grad, x_xpu.grad.to(cpu_device))
