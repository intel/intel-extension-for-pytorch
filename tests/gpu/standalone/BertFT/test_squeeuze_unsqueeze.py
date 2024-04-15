import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_squeeze(self, dtype=torch.float):
        x = torch.randn((2, 384, 1), device=cpu_device, dtype=dtype)
        y_cpu = x.squeeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.squeeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_squeeze_bfloat16(self, dtype=torch.bfloat16):
        x = torch.randn((2, 384, 1), device=cpu_device, dtype=dtype)
        y_cpu = x.squeeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.squeeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_squeeze_float16(self, dtype=torch.float16):
        x = torch.randn((2, 384, 1), device=cpu_device, dtype=dtype)
        y_cpu = x.squeeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.squeeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_unsqueeze_1(self, dtype=torch.float):
        x = torch.randn((2, 384), device=cpu_device, dtype=dtype)
        y_cpu = x.unsqueeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.unsqueeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
    
    def test_unsqueeze_bfloat16_1(self, dtype=torch.bfloat16):
        x = torch.randn((2, 384), device=cpu_device, dtype=dtype)
        y_cpu = x.unsqueeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.unsqueeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
    
    def test_unsqueeze_float16_1(self, dtype=torch.float16):
        x = torch.randn((2, 384), device=cpu_device, dtype=dtype)
        y_cpu = x.unsqueeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.unsqueeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_unsqueeze_2(self, dtype=torch.float):
        x = torch.randn((2, 1, 384), device=cpu_device, dtype=dtype)
        y_cpu = x.unsqueeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.unsqueeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
    
    def test_unsqueeze_bfloat16_2(self, dtype=torch.bfloat16):
        x = torch.randn((2, 1, 384), device=cpu_device, dtype=dtype)
        y_cpu = x.unsqueeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.unsqueeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
    
    def test_unsqueeze_float16_2(self, dtype=torch.float16):
        x = torch.randn((2, 1, 384), device=cpu_device, dtype=dtype)
        y_cpu = x.unsqueeze(1)
        #print("y = ", y_cpu)

        x_xpu = x.to("xpu")
        y_xpu = x_xpu.unsqueeze(1)
        #print("y_xpu ", y_xpu.cpu())

        self.assertEqual(y_cpu, y_xpu.to(cpu_device))
