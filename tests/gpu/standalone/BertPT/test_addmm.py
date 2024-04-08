import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa

xpu_device = torch.device("xpu")
cpu_device = torch.device("cpu")

class TestTorchMethod(TestCase):
    def test_addmm_1(self, dtype=torch.float):
        m1_cpu = torch.randn((1, 1024), dtype=dtype)
        m2_cpu = torch.randn((1024, 2), dtype=dtype)
        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)

        x_cpu = torch.zeros((1, 2), dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        res_cpu = torch.addmm(x_cpu, m1_cpu, m2_cpu)
        res_xpu = torch.addmm(x_xpu, m1_xpu, m2_xpu)
        #print("cpu addmm_ result", res_cpu)
        #print("xpu addmm_ result", res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), rtol=10e-5, atol=10e-5)

    def test_addmm_2(self, dtype=torch.float):
        m1_cpu = torch.rand((1, 1024), dtype=dtype)
        m2_cpu = torch.rand((1024, 1024), dtype=dtype)
        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)

        x_cpu = torch.zeros((1, 1024), dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        res_cpu = torch.addmm(x_cpu, m1_cpu, m2_cpu)
        res_xpu = torch.addmm(x_xpu, m1_xpu, m2_xpu)
        #print("cpu addmm_ result", res_cpu)
        #print("xpu addmm_ result", res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), rtol=10e-5, atol=10e-5)

    def test_addmm_3(self, dtype=torch.float):
        m1_cpu = torch.rand((512, 1024), dtype=dtype)
        m2_cpu = torch.rand((1024, 1024), dtype=dtype)
        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)

        x_cpu = torch.zeros((512, 1024), dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        res_cpu = torch.addmm(x_cpu, m1_cpu, m2_cpu)
        res_xpu = torch.addmm(x_xpu, m1_xpu, m2_xpu)
        #print("cpu addmm_ result", res_cpu)
        #print("xpu addmm_ result", res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), rtol=10e-5, atol=10e-5)

    def test_addmm_4(self, dtype=torch.float):
        m1_cpu = torch.rand((512, 4096), dtype=dtype)
        m2_cpu = torch.rand((4096, 1024), dtype=dtype)
        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)

        x_cpu = torch.zeros((512, 1024), dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        res_cpu = torch.addmm(x_cpu, m1_cpu, m2_cpu)
        res_xpu = torch.addmm(x_xpu, m1_xpu, m2_xpu)
        #print("cpu addmm_ result", res_cpu)
        #print("xpu addmm_ result", res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_addmm_5(self, dtype=torch.float):
        m1_cpu = torch.rand((512, 1024), dtype=dtype)
        m2_cpu = torch.rand((1024, 4096), dtype=dtype)
        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)

        x_cpu = torch.zeros((512, 4096), dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        res_cpu = torch.addmm(x_cpu, m1_cpu, m2_cpu)
        res_xpu = torch.addmm(x_xpu, m1_xpu, m2_xpu)
        #print("cpu addmm_ result", res_cpu)
        #print("xpu addmm_ result", res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), rtol=1e-5, atol=1e-5)

    def test_addmm_6(self, dtype=torch.float):
        m1_cpu = torch.rand((512, 1024), dtype=dtype)
        m2_cpu = torch.rand((1024, 30522), dtype=dtype)
        m1_xpu = m1_cpu.to(xpu_device)
        m2_xpu = m2_cpu.to(xpu_device)

        x_cpu = torch.zeros((512, 30522), dtype=dtype)
        x_xpu = x_cpu.to(xpu_device)
        res_cpu = torch.addmm(x_cpu, m1_cpu, m2_cpu)
        res_xpu = torch.addmm(x_xpu, m1_xpu, m2_xpu)
        #print("cpu addmm_ result", res_cpu)
        #print("xpu addmm_ result", res_xpu.cpu())
        self.assertEqual(res_cpu, res_xpu.cpu(), rtol=1e-5, atol=1e-5)
