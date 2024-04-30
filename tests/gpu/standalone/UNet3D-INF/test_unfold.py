import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

xpu_device = torch.device("xpu")
cpu_device = torch.device("cpu")

class TestTorchMethod(TestCase):
    def test_unfold_1(self, dtype=torch.float):
        x_cpu = torch.randn((128) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_bfloat16_1(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((128) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_unfold_float16_1(self, dtype=torch.float16):
        x_cpu = torch.randn((128) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_2(self, dtype=torch.float):
        x_cpu = torch.randn((256) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_bfloat16_2(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((256) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_unfold_float16_2(self, dtype=torch.float16):
        x_cpu = torch.randn((256) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_3(self, dtype=torch.float):
        x_cpu = torch.randn((32) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_bfloat16_3(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((32) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_unfold_float16_3(self, dtype=torch.float16):
        x_cpu = torch.randn((32) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_4(self, dtype=torch.float):
        x_cpu = torch.randn((320) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_bfloat16_4(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((320) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_unfold_float16_4(self, dtype=torch.float16):
        x_cpu = torch.randn((320) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_5(self, dtype=torch.float):
        x_cpu = torch.randn((64) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())

    def test_unfold_bfloat16_5(self, dtype=torch.bfloat16):
        x_cpu = torch.randn((64) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())
    
    def test_unfold_float16_5(self, dtype=torch.float16):
        x_cpu = torch.randn((64) , device=cpu_device, dtype=dtype, requires_grad=True)
        y_cpu = x_cpu.unfold(0, 2, 1)
        x_xpu = x_cpu.to(xpu_device)
        y_xpu = x_xpu.unfold(0, 2, 1)
        self.assertEqual(y_cpu, y_xpu.cpu())
        