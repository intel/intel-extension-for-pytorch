import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_transpose_1(self, dtype=torch.float):
        x_cpu = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_2(self, dtype=torch.float):
        x_cpu = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_3(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 2), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1024, 2), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((1024, 2), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_4(self, dtype=torch.float):
        x_cpu = torch.randn((2, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((2, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((2, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_5(self, dtype=torch.float):
        x_cpu = torch.randn((512, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((512, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((512, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_6(self, dtype=torch.float):
        x_cpu = torch.randn((512, 4096), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_6(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((512, 4096), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_6(self, dtype=torch.float16):
        x_cpu = torch.randn((512, 4096), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_7(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_7(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1024, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_7(self, dtype=torch.float16):
        x_cpu = torch.randn((1024, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_8(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 4096), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_8(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1024, 4096), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_8(self, dtype=torch.float16):
        x_cpu = torch.randn((1024, 4096), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_9(self, dtype=torch.float):
        x_cpu = torch.randn((4096, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_9(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((4096, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_9(self, dtype=torch.float16):
        x_cpu = torch.randn((4096, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_10(self, dtype=torch.float):
        x_cpu = torch.randn((512, 30522), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_10(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((512, 30522), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_10(self, dtype=torch.float16):
        x_cpu = torch.randn((512, 30522), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_11(self, dtype=torch.float):
        x_cpu = torch.randn((1024, 30522), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_11(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1024, 30522), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_11(self, dtype=torch.float16):
        x_cpu = torch.randn((1024, 30522), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_12(self, dtype=torch.float):
        x_cpu = torch.randn((16, 512, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_12(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((16, 512, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_12(self, dtype=torch.float16):
        x_cpu = torch.randn((16, 512, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_13(self, dtype=torch.float):
        x_cpu = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_13(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_13(self, dtype=torch.float16):
        x_cpu = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_14(self, dtype=torch.float):
        x_cpu = torch.randn((30522, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_14(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((30522, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_14(self, dtype=torch.float16):
        x_cpu = torch.randn((30522, 1024), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_15(self, dtype=torch.float):
        x_cpu = torch.randn((16, 512, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_15(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((16, 512, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_15(self, dtype=torch.float16):
        x_cpu = torch.randn((16, 512, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_16(self, dtype=torch.float):
        x_cpu = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_16(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_16(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_17(self, dtype=torch.float):
        x_cpu = torch.randn((1, 16, 64, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_bfloat16_17(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 16, 64, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_transpose_float16_17(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 16, 64, 512), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_cpu = torch.transpose(x_cpu, 0, 0)
        y_xpu = torch.transpose(x_xpu, 0, 0)
        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
