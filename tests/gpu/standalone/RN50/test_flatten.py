import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes = [
        (1, 2048, 1, 1)
]

class TestTorchMethod(TestCase):
    def test_flatten_with_start_dim(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            user_cpu = torch.randn(shape, device=cpu_device)
            res_cpu = torch.flatten(user_cpu, start_dim=1)
            #print("begin xpu compute:")
            res_xpu = torch.flatten(user_cpu.to("xpu"), start_dim=1)
            #print("xpu result:")
            print(res_xpu.cpu())
            self.assertEqual(res_cpu, res_xpu.cpu())

    def test_flatten(self, dtype=torch.float):
        user_cpu = torch.randn((1, 2048, 1, 1), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu.flatten()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").flatten()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_flatten_bfloat16(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 2048, 1, 1), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu.flatten()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").flatten()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_flatten_float16(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 2048, 1, 1), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu.flatten()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").flatten()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
