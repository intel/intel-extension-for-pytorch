import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes = [
        (1, 2),
        (1024),
        (100),
        (71222),
        (2, 1024),
        (512, 30522),
        (512, 1024),
        (1, 512, 1024),
        (30522, 1024)
]

class TestTorchMethod(TestCase):
    def test_zero_1(self, dtype=torch.float):
        user_cpu = torch.randn((1, 2), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_2(self, dtype=torch.float):
        user_cpu = torch.randn((1024), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_3(self, dtype=torch.float):
        user_cpu = torch.randn((100), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_4(self, dtype=torch.float):
        user_cpu = torch.randn((71222), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_5(self, dtype=torch.float):
        user_cpu = torch.randn((2, 1024), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_6(self, dtype=torch.float):
        user_cpu = torch.randn((512, 30522), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_7(self, dtype=torch.float):
        user_cpu = torch.randn((512, 1024), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_8(self, dtype=torch.float):
        user_cpu = torch.randn((1, 512, 1024), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_zero_9(self, dtype=torch.float):
        user_cpu = torch.randn((30522, 1024), device=cpu_device)
        res_cpu = user_cpu.zero_()
        #print("begin xpu compute:")
        res_xpu = user_cpu.to("xpu").zero_()
        #print("xpu result:")
        #print(res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu())
