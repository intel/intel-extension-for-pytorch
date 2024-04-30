import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_expand_as_1(self, dtype=torch.float):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 128), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_2(self, dtype=torch.float):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 256), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_3(self, dtype=torch.float):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 320), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_4(self, dtype=torch.float):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 32), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_5(self, dtype=torch.float):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())

    def test_expand_as_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        t_cpu = torch.randn((1, 64), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = t_cpu.to(xpu_device)
        self.assertEqual(x_cpu.expand_as(t_cpu), x_xpu.expand_as(t_xpu).cpu())
