import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_transpose_1(self, dtype=torch.float):
        x_cpu = torch.randn((1, 2), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_2(self, dtype=torch.float):
        x_cpu = torch.randn((1, 1024), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_3(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 2), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_4(self, dtype=torch.float):
        x_cpu = torch.randn((2, 1024), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_5(self, dtype=torch.float):
        x_cpu = torch.randn((512, 1024), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_6(self, dtype=torch.float):
        x_cpu = torch.randn((512, 4096), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_7(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 1024), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_8(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 4096), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_9(self, dtype=torch.float):
        x_cpu = torch.randn((4096, 1024), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_10(self, dtype=torch.float):
        x_cpu = torch.randn((512, 30522), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_11(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 30522), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_12(self, dtype=torch.float):
        x_cpu = torch.randn((16, 512, 64), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_13(self, dtype=torch.float):
        x_cpu = torch.randn((16, 64, 512), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_14(self, dtype=torch.float):
        x_cpu = torch.randn((30522, 1024), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_15(self, dtype=torch.float):
        x_cpu = torch.randn((16, 512, 512), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_16(self, dtype=torch.float):
        x_cpu = torch.randn((1, 16, 512, 64), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_17(self, dtype=torch.float):
        x_cpu = torch.randn((1, 16, 64, 512), device=cpu_device)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
