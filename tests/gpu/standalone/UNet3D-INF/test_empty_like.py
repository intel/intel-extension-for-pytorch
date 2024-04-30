import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_empty_like_1(self, dtype=torch.float):
        x_cpu = torch.empty((1, 128, 56, 56, 40), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 128, 56, 56, 40), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_1(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 128, 56, 56, 40), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_2(self, dtype=torch.float):
        x_cpu = torch.empty((1, 256, 28, 28, 20), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 256, 28, 28, 20), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_2(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 256, 28, 28, 20), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_3(self, dtype=torch.float):
        x_cpu = torch.empty((1, 320, 14, 14, 10), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 320, 14, 14, 10), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_3(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 320, 14, 14, 10), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_4(self, dtype=torch.float):
        x_cpu = torch.empty((1, 320, 7, 7, 5), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 320, 7, 7, 5), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_4(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 320, 7, 7, 5), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_5(self, dtype=torch.float):
        x_cpu = torch.empty((1, 32, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 32, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_5(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 32, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_6(self, dtype=torch.float):
        x_cpu = torch.empty((1, 4, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_6(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 4, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_6(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 4, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_7(self, dtype=torch.float):
        x_cpu = torch.empty((1, 64, 112, 112, 80), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_bfloat16_7(self, dtype=torch.bfloat16):
        x_cpu = torch.empty((1, 64, 112, 112, 80), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)

    def test_empty_like_float16_7(self, dtype=torch.float16):
        x_cpu = torch.empty((1, 64, 112, 112, 80), device=cpu_device, dtype=dtype)
        t_cpu = torch.empty_like(x_cpu)
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.empty_like(x_xpu)
        self.assertEqual(t_cpu.shape, t_xpu.shape)
        self.assertEqual(t_cpu.dtype, t_xpu.dtype)