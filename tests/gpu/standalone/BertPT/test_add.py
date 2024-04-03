import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_add_1(self, dtype=torch.float):
        s_cpu = torch.randn((1, 512, 1024), dtype=dtype, device=cpu_device)
        x_cpu = torch.randn((1, 512, 1024), dtype=dtype, device=cpu_device)
        s_xpu = s_cpu.to(dpcpp_device)
        x_xpu = x_cpu.to(dpcpp_device)

        y_cpu = torch.add(x_cpu, s_cpu)
        y_xpu = torch.add(x_xpu, s_xpu)

        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_add_2(self, dtype=torch.float):
        s_cpu = torch.randn((30522, 1024), dtype=dtype, device=cpu_device)
        x_cpu = torch.randn((30522, 1024), dtype=dtype, device=cpu_device)
        s_xpu = s_cpu.to(dpcpp_device)
        x_xpu = x_cpu.to(dpcpp_device)

        y_cpu = torch.add(x_cpu, s_cpu)
        y_xpu = torch.add(x_xpu, s_xpu)

        self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_add_3(self, dtype=torch.float):
        s_cpu = torch.randn((1, 16, 512, 512), dtype=dtype, device=cpu_device)
        x_cpu = torch.randn((1, 1, 1, 512), dtype=dtype, device=cpu_device)
        s_xpu = s_cpu.to(dpcpp_device)
        x_xpu = x_cpu.to(dpcpp_device)

        y_cpu = torch.add(x_cpu, s_cpu)
        y_xpu = torch.add(x_xpu, s_xpu)

        self.assertEqual(y_cpu, y_xpu.cpu())
