import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_unpooling2d(self, dtype=torch.float):
        N, C = 3, 2
        input = torch.randn([N, C, 4, 4], device=cpu_device, dtype=dtype)
        pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        output, indices = pool(input)

        x_cpu = output
        x_dpcpp = output.to("xpu").to(memory_format=torch.channels_last)
        indices_dpcpp = indices.to("xpu")
        grad_cpu = torch.randn([N, C, 4, 4], device=cpu_device)
        grad_dpcpp = grad_cpu.to("xpu")
        output_size = torch.Size([N, C, 5, 5])
        unpool = nn.MaxUnpool2d(2, stride=2)

        x_cpu.requires_grad_(True)
        y_cpu = unpool(x_cpu, indices)
        y_cpu.backward(grad_cpu)

        unpool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = unpool(x_dpcpp, indices_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
