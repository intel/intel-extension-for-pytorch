import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_lshift(self, dtype=torch.float):
        x_cpu = torch.randn(4, dtype=torch.double)
        y_cpu = torch.randn(1, dtype=torch.double)
        x_xpu = x_cpu.to(dpcpp_device)
        y_xpu = y_cpu.to(dpcpp_device)

        re_cpu = x_cpu.__lshift__(y_cpu)
        re_xpu = x_xpu.__lshift__(y_xpu).to(cpu_device)
        self.assertEqual(re_cpu, re_xpu)

        re_cpu = x_cpu.__ilshift__(y_cpu)
        re_xpu = x_xpu.__ilshift__(y_xpu).to(cpu_device)
        self.assertEqual(re_cpu, re_xpu)


    def test_rshift(self, dtype=torch.float):
        x_cpu = torch.randn(4, dtype=torch.double)
        y_cpu = torch.randn(1, dtype=torch.double)
        x_xpu = x_cpu.to(dpcpp_device)
        y_xpu = y_cpu.to(dpcpp_device)

        re_cpu = x_cpu.__rshift__(y_cpu)
        re_xpu = x_xpu.__rshift__(y_xpu).to(cpu_device)
        self.assertEqual(re_cpu, re_xpu)

        re_cpu = x_cpu.__irshift__(y_cpu)
        re_xpu = x_xpu.__irshift__(y_xpu).to(cpu_device)
        self.assertEqual(re_cpu, re_xpu)
