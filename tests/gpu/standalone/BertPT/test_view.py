import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_view_1(self, dtype=torch.float):
        input_cpu = torch.randn(1, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_1(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_1(self, dtype=torch.float16):
        input_cpu = torch.randn(1, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_2(self, dtype=torch.float):
        input_cpu = torch.randn(20, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_2(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(20, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_2(self, dtype=torch.float16):
        input_cpu = torch.randn(20, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())
        
    def test_view_3(self, dtype=torch.float):
        input_cpu = torch.randn(1, 2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_3(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_3(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())
        
    def test_view_4(self, dtype=torch.float):
        input_cpu = torch.randn(1, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_4(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_4(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 1)
        out_dpcpp = input_dpcpp.view(-1, 1)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_5(self, dtype=torch.float):
        input_cpu = torch.randn(1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_5(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_5(self, dtype=torch.float16):
        input_cpu = torch.randn(1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_6(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_6(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_6(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_7(self, dtype=torch.float):
        input_cpu = torch.randn(1, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_7(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_7(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_8(self, dtype=torch.float):
        input_cpu = torch.randn(1, 4096, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_8(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 4096, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_8(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 4096, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_9(self, dtype=torch.float):
        input_cpu = torch.randn(512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_9(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_9(self, dtype=torch.float16):
        input_cpu = torch.randn(512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_10(self, dtype=torch.float):
        input_cpu = torch.randn(512, 4096, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_10(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(512, 4096, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_10(self, dtype=torch.float16):
        input_cpu = torch.randn(512, 4096, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_11(self, dtype=torch.float):
        input_cpu = torch.randn(512, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_11(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(512, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_11(self, dtype=torch.float16):
        input_cpu = torch.randn(512, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_12(self, dtype=torch.float):
        input_cpu = torch.randn(16, 512, 64, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_12(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(16, 512, 64, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_12(self, dtype=torch.float16):
        input_cpu = torch.randn(16, 512, 64, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_13(self, dtype=torch.float):
        input_cpu = torch.randn(16, 64, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_13(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(16, 64, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_13(self, dtype=torch.float16):
        input_cpu = torch.randn(16, 64, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_14(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_14(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_14(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_15(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_15(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_15(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 512, 1024, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_16(self, dtype=torch.float):
        input_cpu = torch.randn(16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_16(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_16(self, dtype=torch.float16):
        input_cpu = torch.randn(16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_17(self, dtype=torch.float):
        input_cpu = torch.randn(16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_17(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_17(self, dtype=torch.float16):
        input_cpu = torch.randn(16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_18(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_18(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 512, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_18(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 512, 30522, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_19(self, dtype=torch.float):
        input_cpu = torch.randn(1, 512, 16, 64, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_19(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 512, 16, 64, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_19(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 512, 16, 64, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_20(self, dtype=torch.float):
        input_cpu = torch.randn(1, 16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_bfloat16_20(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(1, 16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())

    def test_view_float16_20(self, dtype=torch.float16):
        input_cpu = torch.randn(1, 16, 512, 512, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)
        output_cpu = input_cpu.view(-1, 16)
        out_dpcpp = input_dpcpp.view(-1, 16)
        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)
        self.assertEqual(output_cpu, out_dpcpp.cpu())
