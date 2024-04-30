import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa

xpu_device = torch.device("xpu")
cpu_device = torch.device("cpu")

class TestTensorMethod(TestCase):
    def test_as_strided_1(self, dtype=torch.float):
        x_cpu = torch.randn((1, 128, 112, 112, 80), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 128, 112, 112, 80), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 128, 112, 112, 80), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_2(self, dtype=torch.float):
        x_cpu = torch.randn((1, 256, 56, 56, 40), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 256, 56, 56, 40), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 256, 56, 56, 40), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_3(self, dtype=torch.float):
        x_cpu = torch.randn((1, 512, 28, 28, 20), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 512, 28, 28, 20), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 512, 28, 28, 20), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_4(self, dtype=torch.float):
        x_cpu = torch.randn((1, 640, 14, 14, 10), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 640, 14, 14, 10), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 640, 14, 14, 10), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_5(self, dtype=torch.float):
        x_cpu = torch.randn((1, 64, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 64, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 64, 224, 224, 160), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_6(self, dtype=torch.float):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_6(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_6(self, dtype=torch.float16):
        x_cpu = torch.randn((128), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_7(self, dtype=torch.float):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_7(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_7(self, dtype=torch.float16):
        x_cpu = torch.randn((256), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_8(self, dtype=torch.float):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_8(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_8(self, dtype=torch.float16):
        x_cpu = torch.randn((32), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_9(self, dtype=torch.float):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_9(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_9(self, dtype=torch.float16):
        x_cpu = torch.randn((320), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_10(self, dtype=torch.float):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_bfloat16_10(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_as_strided_float16_10(self, dtype=torch.float16):
        x_cpu = torch.randn((64), device=cpu_device, dtype=dtype)
        t_cpu = torch.as_strided(x_cpu, (2, 2), (1, 2))
        x_xpu = x_cpu.to(xpu_device)
        t_xpu = torch.as_strided(x_xpu, (2, 2), (1, 2))
        self.assertEqual(t_cpu, t_xpu.cpu(), rtol=1e-5, atol=1e-5)