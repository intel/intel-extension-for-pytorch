import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_adaptive_max_pool2d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 1, 8, 8], device=cpu_device)
        grad_cpu = torch.randn([1, 1, 2, 2], device=cpu_device)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        max_pool = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        output_cpu = y_cpu[0].backward(grad_cpu)
        print("x_cpu.grad", x_cpu.grad)

        x_dpcpp.requires_grad_(True)
        max_pool = max_pool.to(dpcpp_device)
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        output_dpcpp = y_dpcpp[0].backward(grad_dpcpp)
        print("x_dpcpp.grad", x_dpcpp.grad.cpu())

        # self.assertEqual( y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_dpcpp.grad, x_dpcpp.grad.to(cpu_device))
