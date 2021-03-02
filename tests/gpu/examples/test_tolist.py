import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_tolist(self, dtype=torch.float):
        x_cpu = torch.randn(2, 3, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)

        x_cpu_list = x_cpu.tolist()
        x_xpu_list = x_xpu.tolist()

        print('x_cpu.tolist() = ', x_cpu_list)
        print('x_xpu.tolist() = ', x_xpu_list)

        self.assertEqual(x_cpu_list, x_xpu_list)