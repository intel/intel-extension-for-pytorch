import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_contiguouse(self, dtype=torch.float):
        user_cpu = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu.contiguous()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").contiguous()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_contiguouse_bfloat16(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu.contiguous()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").contiguous()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_contiguouse_float16(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu.contiguous()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").contiguous()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
