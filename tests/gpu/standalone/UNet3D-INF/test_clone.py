import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_clone(self, dtype=torch.float):
        x_cpu = torch.randn((1, 4, 224, 224, 160), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.clone(), x_xpu.clone().to(cpu_device))

    def test_clone_bfloat16(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 4, 224, 224, 160), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.clone(), x_xpu.clone().to(cpu_device))

    def test_clone_float16(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 4, 224, 224, 160), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.clone(), x_xpu.clone().to(cpu_device))
