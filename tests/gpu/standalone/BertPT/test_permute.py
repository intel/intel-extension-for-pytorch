import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_permute_1(self, dtype=torch.float):
        user_cpu = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        res_cpu = torch.permute(user_cpu, (2, 0, 1, 3))
        #print("begin xpu compute:")
        res_xpu = torch.permute(user_cpu.to("xpu"), (2, 0, 1, 3))
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_permute_bfloat16_1(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        res_cpu = torch.permute(user_cpu, (2, 0, 1, 3))
        #print("begin xpu compute:")
        res_xpu = torch.permute(user_cpu.to("xpu"), (2, 0, 1, 3))
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_permute_float16_1(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        res_cpu = torch.permute(user_cpu, (2, 0, 1, 3))
        #print("begin xpu compute:")
        res_xpu = torch.permute(user_cpu.to("xpu"), (2, 0, 1, 3))
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_permute_2(self, dtype=torch.float):
        user_cpu = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        res_cpu = torch.permute(user_cpu, (2, 0, 1, 3))
        #print("begin xpu compute:")
        res_xpu = torch.permute(user_cpu.to("xpu"), (2, 0, 1, 3))
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_permute_bfloat16_2(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        res_cpu = torch.permute(user_cpu, (2, 0, 1, 3))
        #print("begin xpu compute:")
        res_xpu = torch.permute(user_cpu.to("xpu"), (2, 0, 1, 3))
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_permute_float16_2(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        res_cpu = torch.permute(user_cpu, (2, 0, 1, 3))
        #print("begin xpu compute:")
        res_xpu = torch.permute(user_cpu.to("xpu"), (2, 0, 1, 3))
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
