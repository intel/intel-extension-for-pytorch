import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_ne(self, dtype=torch.float):
        x_cpu = torch.randn(512, device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.ne(x_cpu, 2)
        y_xpu = torch.ne(x_xpu, 2)

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_ne_bfloat16(self, dtype=torch.bfloat16):
        x_cpu = torch.randn(512, device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.ne(x_cpu, 2)
        y_xpu = torch.ne(x_xpu, 2)

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_ne_float16(self, dtype=torch.float16):
        x_cpu = torch.randn(512, device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.ne(x_cpu, 2)
        y_xpu = torch.ne(x_xpu, 2)

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
