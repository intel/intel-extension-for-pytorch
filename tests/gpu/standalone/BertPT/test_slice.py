import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_slice_1(self, dtype=torch.float):
        user_cpu = torch.randn((398), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_1(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((398), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_1(self, dtype=torch.float16):
        user_cpu = torch.randn((398), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_2(self, dtype=torch.float):
        user_cpu = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_2(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_2(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_3(self, dtype=torch.float):
        user_cpu = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_3(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_3(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_4(self, dtype=torch.float):
        user_cpu = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_bfloat16_4(self, dtype=torch.bfloat16):
        user_cpu = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_slice_float16_4(self, dtype=torch.float16):
        user_cpu = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        res_cpu = user_cpu[0]
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu")[0]
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
