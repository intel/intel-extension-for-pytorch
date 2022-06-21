import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import numpy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# functionality


class TestNNMethod(TestCase):
    def test_LeakyReLU(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn([1, 2, 3, 4], device=cpu_device, requires_grad=True)
        grad_x = torch.randn(1, 2, 3, 4, device=cpu_device, requires_grad=True)
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        print("cpu output ", y_cpu)
        print("cpu grad ", x_cpu.grad)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu"), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        print("dpcpp output", y_dpcpp.cpu())
        print("dpcpp grad ", x_dpcpp.grad.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_LeakyReLU_channels_last_fwd(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn([1, 2, 3, 4], device=cpu_device, requires_grad=True)
        y_cpu = Xelu(x_cpu)
        print("cpu output ", y_cpu)

        Xelu.to("xpu")
        x_dpcpp = x_cpu.to("xpu").to(memory_format=torch.channels_last)
        y_dpcpp = Xelu(x_dpcpp)

        print("dpcpp output", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

    def test_LeakyReLU_channels_last(self, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
        x_cpu = torch.randn([1, 2, 3, 4], device=cpu_device, requires_grad=True)
        grad_x = torch.randn(1, 2, 3, 4, device=cpu_device, requires_grad=True)
        y_cpu = Xelu(x_cpu)
        y_cpu.backward(grad_x)

        print("cpu output ", y_cpu)
        print("cpu grad ", x_cpu.grad)

        Xelu.to("xpu")
        Xelu.zero_grad()

        x_dpcpp = Variable(x_cpu.to("xpu").to(memory_format=torch.channels_last), requires_grad=True)
        grad_dpcpp = Variable(grad_x.to("xpu").to(memory_format=torch.channels_last), requires_grad=True)

        y_dpcpp = Xelu(x_dpcpp)
        y_dpcpp.backward(grad_dpcpp)

        print("dpcpp output", y_dpcpp.cpu())
        print("dpcpp grad ", x_dpcpp.grad.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())

    def test_inplace_LeakyReLU_channels_last_fwd(self, Xelu=nn.LeakyReLU(0.1, inplace=True), dtype=torch.float):
        x_cpu = torch.randn([1, 2, 3, 4], device=cpu_device)
        ref_cpu = x_cpu.detach().clone()
        y_cpu = Xelu(x_cpu)
        print("cpu output ", y_cpu)

        Xelu.to("xpu")
        x_dpcpp = ref_cpu.to("xpu").to(memory_format=torch.channels_last)
        y_dpcpp = Xelu(x_dpcpp)

        print("dpcpp output", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
