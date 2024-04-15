import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_simple_copy_1(self, dtype=torch.float):
        src = torch.randn((1), device=cpu_device, dtype=dtype)
        dst = torch.randn((1), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_1(self, dtype=torch.bfloat16):
        src = torch.randn((1), device=cpu_device, dtype=dtype)
        dst = torch.randn((1), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_1(self, dtype=torch.float16):
        src = torch.randn((1), device=cpu_device, dtype=dtype)
        dst = torch.randn((1), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))
    
    def test_simple_copy_2(self, dtype=torch.float):
        src = torch.randn((20), device=cpu_device, dtype=dtype)
        dst = torch.randn((20), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_2(self, dtype=torch.bfloat16):
        src = torch.randn((20), device=cpu_device, dtype=dtype)
        dst = torch.randn((20), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_2(self, dtype=torch.float16):
        src = torch.randn((20), device=cpu_device, dtype=dtype)
        dst = torch.randn((20), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))
    
    def test_simple_copy_3(self, dtype=torch.float):
        src = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_3(self, dtype=torch.bfloat16):
        src = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_3(self, dtype=torch.float16):
        src = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_4(self, dtype=torch.float):
        src = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_4(self, dtype=torch.bfloat16):
        src = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_4(self, dtype=torch.float16):
        src = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_5(self, dtype=torch.float):
        src = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_5(self, dtype=torch.bfloat16):
        src = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_5(self, dtype=torch.float16):
        src = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_6(self, dtype=torch.float):
        src = torch.randn((512, 32522), device=cpu_device, dtype=dtype)
        dst = torch.randn((512, 32522), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_6(self, dtype=torch.bfloat16):
        src = torch.randn((512, 32522), device=cpu_device, dtype=dtype)
        dst = torch.randn((512, 32522), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_6(self, dtype=torch.float16):
        src = torch.randn((512, 32522), device=cpu_device, dtype=dtype)
        dst = torch.randn((512, 32522), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_7(self, dtype=torch.float):
        src = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_7(self, dtype=torch.bfloat16):
        src = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_7(self, dtype=torch.float16):
        src = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 1, 1, 512), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_8(self, dtype=torch.float):
        src = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_8(self, dtype=torch.bfloat16):
        src = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_8(self, dtype=torch.float16):
        src = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512, 1024), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_9(self, dtype=torch.float):
        src = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_bfloat16_9(self, dtype=torch.bfloat16):
        src = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))

    def test_simple_copy_float16_9(self, dtype=torch.float16):
        src = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        dst = torch.randn((1, 512, 16, 64), device=cpu_device, dtype=dtype)
        dst.copy_(src)
        src_dpcpp = src.to(dpcpp_device)
        dst_dpcpp = dst.to(dpcpp_device)
        dst_dpcpp.copy_(src_dpcpp)
        self.assertEqual(dst, dst_dpcpp.to(cpu_device))
