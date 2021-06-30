import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_avg_pool3d(self, dtype=torch.float):
        x_cpu = torch.ones([8, 8, 24, 24], device=cpu_device)
        grad_cpu = torch.ones([8, 8, 24, 24], device=cpu_device)

        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)
        y_cpu.backward(torch.ones([8, 8, 24, 24], device=cpu_device))
        # print("y_cpu backward", x_cpu.grad)

        x_dpcpp = torch.ones([8, 8, 24, 24], device=dpcpp_device,)
        x_dpcpp.requires_grad_(True)
        y_dpcpp = avg_pool(x_dpcpp)

        # print("y_dpcpp", y_dpcpp.cpu())

        # grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp.backward(torch.ones([8, 8, 24, 24], device=dpcpp_device))
        # print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.ones([8, 8, 24, 24, 24], device=cpu_device)
        x_dpcpp = torch.ones([8, 8, 24, 24, 24], device=dpcpp_device).to(memory_format=torch.channels_last_3d)
        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)

        y_dpcpp = avg_pool(x_dpcpp)
        # print("y_dpcpp", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.ones([8, 8, 8, 8, 8], device=cpu_device)
        grad_cpu = torch.ones([8, 8, 8, 8, 8], device=cpu_device)
        x_dpcpp = torch.ones([8, 8, 8, 8, 8], device=dpcpp_device).to(memory_format=torch.channels_last_3d)
        grad_dpcpp = grad_cpu.to(dpcpp_device)
        avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = avg_pool(x_cpu)
        # print("y_cpu", y_cpu)
        y_cpu.backward(grad_cpu)
        # print("y_cpu backward", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        y_dpcpp = avg_pool(x_dpcpp)

        # print("y_dpcpp", y_dpcpp.cpu())
        y_dpcpp.backward(grad_dpcpp)
        # print("y_dpcpp backward", x_dpcpp.grad.cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))
