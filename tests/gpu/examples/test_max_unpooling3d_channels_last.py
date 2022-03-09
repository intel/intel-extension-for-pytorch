import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_max_unpooling3d(self, dtype=torch.float):
        N, C, D, H, W = 2, 3, 4, 4, 4
        input = torch.randn([N, C, D, H, W], device=cpu_device, dtype=dtype)
        pool = nn.MaxPool3d(2, stride=2, return_indices=True)
        output, indices = pool(input)

        x_cpu = output
        x_dpcpp = output.to("xpu").to(memory_format=torch.channels_last_3d)
        indices_dpcpp = indices.to("xpu").to(memory_format=torch.channels_last_3d)
        grad_cpu = torch.randn([N, C, D, H, W], device=cpu_device)
        grad_dpcpp = grad_cpu.to("xpu")
        unpool = nn.MaxUnpool3d(2, stride=2)

        x_cpu.requires_grad_(True)
        y_cpu = unpool(x_cpu, indices)
        y_cpu.backward(grad_cpu)

        unpool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = unpool(x_dpcpp, indices_dpcpp)
        y_dpcpp.backward(grad_dpcpp)
        self.assertEqual(y_cpu[0], y_dpcpp[0].cpu())
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
