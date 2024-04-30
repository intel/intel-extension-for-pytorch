import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_reshape_1(self, dtype=torch.float):
        x_cpu = torch.randn((1, 128, 56, 56, 40), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 128, 56, 56, 40), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 128, 56, 56, 40), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_2(self, dtype=torch.float):
        x_cpu = torch.randn((1, 256, 28, 28, 20), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 256, 28, 28, 20), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 256, 28, 28, 20), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_3(self, dtype=torch.float):
        x_cpu = torch.randn((1, 320, 14, 14, 10), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 320, 14, 14, 10), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 320, 14, 14, 10), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_4(self, dtype=torch.float):
        x_cpu = torch.randn((1, 320, 7, 7, 5), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 320, 7, 7, 5), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 320, 7, 7, 5), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_5(self, dtype=torch.float):
        x_cpu = torch.randn((1, 32, 224, 224, 160), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 32, 224, 224, 160), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 32, 224, 224, 160), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_6(self, dtype=torch.float):
        x_cpu = torch.randn((1, 64, 112, 112, 80), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_bfloat16_6(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((1, 64, 112, 112, 80), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))

    def test_reshape_float16_6(self, dtype=torch.float16):
        x_cpu = torch.randn((1, 64, 112, 112, 80), device=cpu_device, dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        res_cpu = torch.reshape(x_cpu, (-1,))
        res_xpu = torch.reshape(x_xpu, (-1,))
        self.assertEqual(res_cpu, res_xpu.to(cpu_device))
        
