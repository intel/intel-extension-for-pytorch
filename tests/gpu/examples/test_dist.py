import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_dist(self, dtype=torch.float32):
        x_cpu = torch.randn(4)
        y_cpu = torch.randn(4)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = y_cpu.to(dpcpp_device)

        z1_cpu = torch.dist(x_cpu, y_cpu, 3.5)
        z1_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 3.5)
        print("z1_cpu = ", z1_cpu)
        print("z1_dpcpp = ", z1_dpcpp)
        self.assertEqual(z1_cpu, z1_dpcpp.to(cpu_device))

        z2_cpu = torch.dist(x_cpu, y_cpu, 3)
        z2_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 3)
        print("z2_cpu = ", z2_cpu)
        print("z2_dpcpp = ", z2_dpcpp)
        self.assertEqual(z2_cpu, z2_dpcpp.to(cpu_device))

        z3_cpu = torch.dist(x_cpu, y_cpu, 0)
        z3_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 0)
        print("z3_cpu = ", z3_cpu)
        print("z3_dpcpp = ", z3_dpcpp)
        self.assertEqual(z3_cpu, z3_dpcpp.to(cpu_device))

        z4_cpu = torch.dist(x_cpu, y_cpu, 1)
        z4_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 1)
        print("z4_cpu = ", z4_cpu)
        print("z4_dpcpp = ", z4_dpcpp)
        self.assertEqual(z4_cpu, z4_dpcpp.to(cpu_device))

        self.assertEqual(x_cpu.dist(y_cpu, 3.5), x_dpcpp.dist(y_dpcpp, 3.5).to(cpu_device))
        self.assertEqual(x_cpu.dist(y_cpu, 3), x_dpcpp.dist(y_dpcpp, 3).to(cpu_device))
        self.assertEqual(x_cpu.dist(y_cpu, 0), x_dpcpp.dist(y_dpcpp, 0).to(cpu_device))
        self.assertEqual(x_cpu.dist(y_cpu, 1), x_dpcpp.dist(y_dpcpp, 1).to(cpu_device))