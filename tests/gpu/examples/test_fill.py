import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_fill(self, dtype=torch.float):
        x_cpu = torch.ones([3, 5, 7, 9], device=cpu_device)
        y_cpu = x_cpu.fill_(2)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = x_dpcpp.fill_(2)
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

    def test_fill_bool(self, dtype=torch.bool):
        x_cpu = torch.ones([2, 4, 6, 8], dtype=torch.bool, device=cpu_device)
        y_cpu = x_cpu.fill_(True)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = x_dpcpp.fill_(True)
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

    def test_fill_int8(self, dtype=torch.int8):
        x_cpu = torch.ones([3, 5, 7, 9], dtype=torch.int8, device=cpu_device)
        y_cpu = x_cpu.fill_(-5)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = x_dpcpp.fill_(-5)
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

    def test_fill_uint8(self, dtype=torch.uint8):
        x_cpu = torch.ones([2, 4, 6, 8], dtype=torch.uint8, device=cpu_device)
        y_cpu = x_cpu.fill_(15)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = x_dpcpp.fill_(15)
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
