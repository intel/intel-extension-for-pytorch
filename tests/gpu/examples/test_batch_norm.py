import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_batch_norm_half(self, dtype=torch.half):
        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)

        bn = nn.BatchNorm2d(2)
        y_cpu = bn(x_i)
        bn.to(dpcpp_device).to(dtype)
        y_dpcpp = bn(x_dpcpp_i)
        self.assertEqual(y_cpu, y_dpcpp.cpu().float(), atol=1e-2, rtol=0)

    def test_batch_norm_bfloat16(self, dtype=torch.bfloat16):
        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        x_dpcpp_i = x_i.to(dpcpp_device).to(dtype)

        bn = nn.BatchNorm2d(2)
        y_cpu = bn(x_i)
        bn.to(dpcpp_device).to(dtype)
        y_dpcpp = bn(x_dpcpp_i)
        self.assertEqual(y_cpu, y_dpcpp.cpu().float(), atol=1e-1, rtol=0)

    def test_batch_norm(self, dtype=torch.float):

        x_i = torch.randn([2, 2, 3, 3], device=cpu_device)
        grad_i = torch.randn([2, 2, 3, 3], device=cpu_device)

        x_dpcpp_i = x_i.to(dpcpp_device)
        grad_dpcpp_i = grad_i.to(dpcpp_device)

        self.assertEqual(x_i, x_dpcpp_i.to(cpu_device))
        self.assertEqual(grad_i, grad_dpcpp_i.to(cpu_device))

        x_cpu = Variable(x_i, requires_grad=True)
        grad_cpu = Variable(grad_i, requires_grad=True)
        bn1 = nn.BatchNorm2d(2)
        bn2 = nn.BatchNorm2d(2)
        y_cpu1 = bn1(x_cpu)
        y_cpu = bn2(y_cpu1)

        y_cpu.backward(grad_cpu)

        print("x_cpu = ", y_cpu)
        print("x_cpu.grad = ", x_cpu.grad)

        x_dpcpp = Variable(x_dpcpp_i, requires_grad=True)
        grad_dpcpp = Variable(grad_dpcpp_i, requires_grad=True)
        bn1.to(dpcpp_device)
        bn2.to(dpcpp_device)

        y_dpcpp1 = bn1(x_dpcpp)
        y_dpcpp = bn2(y_dpcpp1)

        y_dpcpp.backward(grad_dpcpp)

        #y = y_dpcpp1.cpu()
        #y = Variable(y, requires_grad = True)
        # y.backward(grad_cpu)
        print("y_dpcpp = ", y_dpcpp.cpu())
        print("x_dpcpp.grad", x_dpcpp.grad.cpu())
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))
