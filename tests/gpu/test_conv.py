import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_conv(self, dtype=torch.float):
        # functionality
        x_cpu = torch.ones([1, 2, 3, 3], device=cpu_device)
        x_dpcpp = torch.ones([1, 2, 3, 3], device=dpcpp_device)
        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))

        conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)

        grad_cpu = torch.ones([1, 2, 3, 3], device=cpu_device)
        grad_dpcpp = torch.ones([1, 2, 3, 3], device=dpcpp_device)
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        x_cpu.requires_grad_(True)
        y_cpu = conv1(x_cpu)
        conv1.zero_grad()
        output_cpu = y_cpu.backward(grad_cpu)
        print("ref: ")
        print(y_cpu)
        print("ref grad: ")
        print(x_cpu.grad[0])

        x_dpcpp.requires_grad_(True)
        conv1.to(dpcpp_device)
        y_dpcpp = conv1(x_dpcpp)
        conv1.zero_grad()
        output_dpcpp = y_dpcpp.backward(grad_dpcpp)
        print("real: ")
        print(y_dpcpp.to(cpu_device))
        print("real grad: ")
        print(x_dpcpp.grad[0].to(cpu_device))

        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad[0], x_dpcpp.grad[0].to(cpu_device))

        x_cpu_2 = torch.randn([1, 2, 2, 1, 1], device=cpu_device,
                              dtype=dtype, requires_grad=True)
        grad = torch.ones([1, 2, 2, 1, 1], device=cpu_device,
                          dtype=dtype, requires_grad=True)
        conv3 = nn.Conv3d(2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        y_cpu_2 = conv3(x_cpu_2)
        y_cpu_2.backward(grad)

        conv3.to(dpcpp_device)
        x_dpcpp_2 = x_cpu_2.to(dpcpp_device)
        y_dpcpp_2 = conv3(x_dpcpp_2)
        grad_dpcpp = grad.to(dpcpp_device)

        y_dpcpp_2.backward(grad_dpcpp)

        print("ref: ")
        print(y_dpcpp_2)
        print("ref backward: ")
        print(x_dpcpp_2)

        print("real: ")
        print(y_dpcpp_2.to(cpu_device))
        print("real backward: ")
        print(x_dpcpp_2.to(cpu_device))
        self.assertEqual(x_cpu_2, x_dpcpp_2)
        self.assertEqual(grad, grad_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu_2, y_dpcpp_2.to(cpu_device))
