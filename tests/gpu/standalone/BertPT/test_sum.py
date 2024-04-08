import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_sum_1(self, dtype=torch.float):
        user_cpu = torch.randn((1), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 0, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 0, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_sum_2(self, dtype=torch.float):
        user_cpu = torch.randn((20), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 0, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 0, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_sum_3(self, dtype=torch.float):
        user_cpu = torch.randn((1, 2), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 1, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 1, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_sum_4(self, dtype=torch.float):
        user_cpu = torch.randn((1, 1024), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 1, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 1, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_sum_5(self, dtype=torch.float):
        user_cpu = torch.rand((512, 1024), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 1, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 1, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_sum_6(self, dtype=torch.float):
        user_cpu = torch.rand((512, 4096), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 1, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 1, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_sum_7(self, dtype=torch.float):
        user_cpu = torch.rand((512, 30522), device=cpu_device)
        res_cpu = torch.sum(user_cpu, 1, True)
        #print("begin xpu compute:")
        res_xpu = torch.sum(user_cpu.to("xpu"), 1, True)
        # print("xpu result:")
        # print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
