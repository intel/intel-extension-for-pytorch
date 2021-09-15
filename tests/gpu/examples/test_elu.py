import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

import ipex

import numpy

dtype = torch.float32
cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_elu(self, dtype=torch.float):
        def test_Xelu(Xelu):
            x_cpu = torch.randn([3, 4], device=cpu_device, dtype=dtype, requires_grad=True)
            grad_x = torch.ones(3, 4, device=cpu_device, dtype=dtype, requires_grad=True)

            y_cpu = Xelu(x_cpu)
            y_cpu.backward(grad_x)

            print("cpu output ", y_cpu)
            print("cpu grad ", x_cpu.grad)

            xelu_dpcpp = Xelu.to("xpu")

            x_dpcpp = Variable(x_cpu.clone().to("xpu"), requires_grad=True)
            grad_dpcpp = Variable(grad_x.to("xpu"), requires_grad=True)

            y_dpcpp = xelu_dpcpp(x_dpcpp)
            y_dpcpp.backward(grad_dpcpp)

            print("dpcpp output", y_dpcpp.cpu())
            print("dpcpp grad ", x_dpcpp.grad.cpu())

            self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
            self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

        test_Xelu(nn.ELU())
        test_Xelu(nn.CELU())
        test_Xelu(nn.SELU())
