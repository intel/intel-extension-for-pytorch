import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_slice_1(self, dtype=torch.float):
        user_cpu = torch.randn((1, 128, 112, 112, 80), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_1(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 128, 112, 112, 80), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_1(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 128, 112, 112, 80), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_2(self, dtype=torch.float):
        user_cpu = torch.randn((1, 256, 56, 56, 40), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_2(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 256, 56, 56, 40), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_2(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 256, 56, 56, 40), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_3(self, dtype=torch.float):
        user_cpu = torch.randn((1, 512, 28, 28, 20), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_3(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 512, 28, 28, 20), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_3(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 512, 28, 28, 20), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_4(self, dtype=torch.float):
        user_cpu = torch.randn((1, 640, 14, 14, 10), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_4(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 640, 14, 14, 10), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_4(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 640, 14, 14, 10), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_5(self, dtype=torch.float):
        user_cpu = torch.randn((1, 64, 224, 224, 160), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_5(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 64, 224, 224, 160), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_5(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 64, 224, 224, 160), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0][5][5]
        res_xpu = user_cpu.to("xpu")[0][5][5]
        self.assertEqual(res_cpu, res_xpu.cpu())
