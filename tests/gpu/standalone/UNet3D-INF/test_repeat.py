import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_repeat_1(self, dtype=torch.float):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(128, 2, 2), x_xpu.repeat(128, 2, 2).to(cpu_device))

    def test_repeat_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(128, 2, 2), x_xpu.repeat(128, 2, 2).to(cpu_device))

    def test_repeat_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(128, 2, 2), x_xpu.repeat(128, 2, 2).to(cpu_device))

    def test_repeat_2(self, dtype=torch.float):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(256, 2, 2), x_xpu.repeat(256, 2, 2).to(cpu_device))

    def test_repeat_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(256, 2, 2), x_xpu.repeat(256, 2, 2).to(cpu_device))

    def test_repeat_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(256, 2, 2), x_xpu.repeat(256, 2, 2).to(cpu_device))

    def test_repeat_3(self, dtype=torch.float):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(32, 2, 2), x_xpu.repeat(32, 2, 2).to(cpu_device))

    def test_repeat_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(32, 2, 2), x_xpu.repeat(32, 2, 2).to(cpu_device))

    def test_repeat_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(32, 2, 2), x_xpu.repeat(32, 2, 2).to(cpu_device))

    def test_repeat_4(self, dtype=torch.float):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(320, 2, 2), x_xpu.repeat(320, 2, 2).to(cpu_device))

    def test_repeat_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(320, 2, 2), x_xpu.repeat(320, 2, 2).to(cpu_device))

    def test_repeat_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(320, 2, 2), x_xpu.repeat(320, 2, 2).to(cpu_device))

    def test_repeat_5(self, dtype=torch.float):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(64, 2, 2), x_xpu.repeat(64, 2, 2).to(cpu_device))

    def test_repeat_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(64, 2, 2), x_xpu.repeat(64, 2, 2).to(cpu_device))

    def test_repeat_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        self.assertEqual(x_cpu.repeat(64, 2, 2), x_xpu.repeat(64, 2, 2).to(cpu_device))