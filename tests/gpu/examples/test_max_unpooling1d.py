import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_unpooling1d(self, dtype=torch.float):

        input = torch.randn([2, 2, 4], device=cpu_device, dtype=dtype)
        pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        output, indices = pool(input)

        x_cpu = output
        x_dpcpp = output.to("xpu")
        indices_dpcpp = indices.to("xpu")
        grad_cpu = torch.randn([2, 2, 4], device=cpu_device)
        grad_dpcpp = grad_cpu.to("xpu")
        unpool = nn.MaxUnpool1d(2, stride=2)

        x_cpu.requires_grad_(True)
        y_cpu = unpool(x_cpu, indices)
        print("y_cpu", y_cpu)
        y_cpu.backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        unpool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = unpool(x_dpcpp, indices_dpcpp)
        print("y_dpcpp", y_dpcpp.to("cpu"))
        y_dpcpp.backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.to("cpu"))
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
