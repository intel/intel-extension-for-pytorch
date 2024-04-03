import torch
import torch.nn.functional
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import copy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_activation_gelu_1(self, dtype=torch.float):
        C, H, W = 1, 512, 1024
        GELU = torch.nn.GELU()
        GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
        x_cpu = torch.randn([C, H, W], dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = GELU(x_cpu)
        y_dpcpp = GELU_dpcpp(x_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())
    
    def test_activation_gelu_2(self, dtype=torch.float):
        C, H, W = 1, 512, 4096
        GELU = torch.nn.GELU()
        GELU_dpcpp = copy.deepcopy(GELU).to("xpu")
        x_cpu = torch.randn([C, H, W], dtype=dtype)
        x_dpcpp = x_cpu.to("xpu")
        x_cpu.requires_grad_(True)
        x_dpcpp.requires_grad_(True)
        y_cpu = GELU(x_cpu)
        y_dpcpp = GELU_dpcpp(x_dpcpp)

        self.assertEqual(y_cpu, y_dpcpp.cpu())
        y_cpu.backward(x_cpu)
        y_dpcpp.backward(x_dpcpp)

        self.assertEqual(x_cpu.grad, x_dpcpp.grad.cpu())


