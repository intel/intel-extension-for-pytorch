import numpy
import torch
import torch.nn as nn
import torch_ipex
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

# functionality
x_cpu = torch.randn([3, 4], device=cpu_device, requires_grad=True)
grad_x = torch.randn(3, 4, device=cpu_device, requires_grad=True)


class TestNNMethod(TestCase):
    def test_LeakReLU(self, x_cpu=x_cpu, grad_x=grad_x, Xelu=nn.LeakyReLU(0.1), dtype=torch.float):
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
