import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_fill(self, dtype=torch.float):
        x_cpu = torch.ones([1, 2, 3, 4], device=cpu_device)
        y_cpu = x_cpu.fill_(2)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = x_dpcpp.fill_(2)
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
# print(y_cpu)

# x_dpcpp1 = x_cpu1.to("xpu")
#
#
# y_dpcpp = x_dpcpp1[1].fill_(2)
# print("dpcpp:", y_dpcpp.cpu())
