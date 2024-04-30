import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_expand_1(self, dtype=torch.float):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_2(self, dtype=torch.float):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_3(self, dtype=torch.float):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_4(self, dtype=torch.float):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_5(self, dtype=torch.float):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())

    def test_expand_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand(3, -1), x_xpu.expand(3, -1).cpu())
