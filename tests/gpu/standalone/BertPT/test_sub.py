import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_sub(self, dtype=torch.float):
        x_cpu = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        Other = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        Other_xpu = Other.to("xpu")
        self.assertEqual(torch.sub(x_cpu, 1), torch.sub(x_xpu, 1).to(cpu_device))
        self.assertEqual(torch.sub(x_cpu, Other), torch.sub(x_xpu, Other_xpu).to(cpu_device))

    def test_sub_bfloat16(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        Other = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        Other_xpu = Other.to("xpu")
        self.assertEqual(torch.sub(x_cpu, 1), torch.sub(x_xpu, 1).to(cpu_device))
        self.assertEqual(torch.sub(x_cpu, Other), torch.sub(x_xpu, Other_xpu).to(cpu_device))

    def test_sub_float16(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        Other = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        Other_xpu = Other.to("xpu")
        self.assertEqual(torch.sub(x_cpu, 1), torch.sub(x_xpu, 1).to(cpu_device))
        self.assertEqual(torch.sub(x_cpu, Other), torch.sub(x_xpu, Other_xpu).to(cpu_device))
