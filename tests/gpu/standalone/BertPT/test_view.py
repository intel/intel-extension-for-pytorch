import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes_1_d = [
            (1),
            (20),
            (1, 2),
            (1, 30522)
            ]
shapes = [
        (1024),
        (1, 512),
        (1, 1024),
        (1, 4096),
        (512, 1024),
        (512, 4096),
        (512, 30522),
        (16, 512, 64),
        (16, 64, 512),
        (1, 512, 1024),
        (1, 512, 1024),
        (16, 512, 512),
        (1, 512, 30522),
        (1, 512, 16, 64),
        (1, 16, 512, 512)
        ]

class TestTensorMethod(TestCase):
    def test_view_1(self, dtype=torch.float):
        input_cpu = torch.randn(1)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_2(self, dtype=torch.float):
        input_cpu = torch.randn(20)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())
        
    def test_view_3(self, dtype=torch.float):
        input_cpu = torch.randn(1, 2)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())
        
    def test_view_4(self, dtype=torch.float):
        input_cpu = torch.randn(1, 30522)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_5(self, dtype=torch.float):
        input_cpu = torch.randn(1024)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_6(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_7(self, dtype=torch.float):
        input_cpu = torch.randn(1, 1024)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_8(self, dtype=torch.float):
        input_cpu = torch.randn(1, 4096)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_9(self, dtype=torch.float):
        input_cpu = torch.randn(512, 1024)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_10(self, dtype=torch.float):
        input_cpu = torch.randn(512, 4096)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_11(self, dtype=torch.float):
        input_cpu = torch.randn(512, 30522)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_12(self, dtype=torch.float):
        input_cpu = torch.randn(16, 512, 64)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_13(self, dtype=torch.float):
        input_cpu = torch.randn(16, 64, 512)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_14(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 1024)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_15(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 1024)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_16(self, dtype=torch.float):
        input_cpu = torch.randn(16, 512, 512)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_17(self, dtype=torch.float):
        input_cpu = torch.randn(16, 512, 512)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_18(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 30522)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_19(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 16, 64)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_20(self, dtype=torch.float):
        input_cpu = torch.randn(1, 16, 512, 512)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())
