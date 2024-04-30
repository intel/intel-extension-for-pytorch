import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_narrow_1(self, dtype=torch.float):
        x_cpu = torch.randn((1, 128, 112, 112, 80), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 128, 112, 112, 80), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 128, 112, 112, 80), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_2(self, dtype=torch.float):
        x_cpu = torch.randn((1, 256, 56, 56, 40), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 256, 56, 56, 40), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 256, 56, 56, 40), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_3(self, dtype=torch.float):
        x_cpu = torch.randn((1, 512, 28, 28, 20), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 512, 28, 28, 20), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 512, 28, 28, 20), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_4(self, dtype=torch.float):
        x_cpu = torch.randn((1, 640, 14, 14, 10), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 640, 14, 14, 10), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 640, 14, 14, 10), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_5(self, dtype=torch.float):
        x_cpu = torch.randn((1, 64, 224, 224, 160), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 64, 224, 224, 160), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())

    def test_narrow_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 64, 224, 224, 160), dtype=dtype)
        x_cpu.narrow(2, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        x_xpu.narrow(2, 2, 1)
        self.assertEqual(x_cpu, x_xpu.cpu())
