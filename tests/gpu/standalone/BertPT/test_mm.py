import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_mm_1(self, dtype=torch.float):
        batch1 = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((2, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_mm_bfloat16_1(self, dtype=torch.bfloat16):
        batch1 = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((2, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_mm_float16_1(self, dtype=torch.bfloat16):
        batch1 = torch.randn((1, 2), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((2, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
        )

    def test_mm_2(self, dtype=torch.float):
        batch1 = torch.randn((2, 1), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_mm_bfloat16_2(self, dtype=torch.bfloat16):
        batch1 = torch.randn((2, 1), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_mm_float16_2(self, dtype=torch.bfloat16):
        batch1 = torch.randn((2, 1), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
        )
    
    def test_mm_3(self, dtype=torch.float):
        batch1 = torch.randn((1024, 1), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_mm_bfloat16_3(self, dtype=torch.bfloat16):
        batch1 = torch.randn((1024, 1), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_mm_float16_3(self, dtype=torch.bfloat16):
        batch1 = torch.randn((1024, 1), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
        )

    def test_mm_4(self, dtype=torch.float):
        batch1 = torch.rand((1, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-5, atol=10e-5
        )

    def test_mm_bfloat16_4(self, dtype=torch.bfloat16):
        batch1 = torch.rand((1, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-5, atol=10e-5
        )

    def test_mm_float16_4(self, dtype=torch.bfloat16):
        batch1 = torch.rand((1, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=10e-3, atol=10e-3
        )

    def test_mm_5(self, dtype=torch.float):
        batch1 = torch.rand((1024, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_5(self, dtype=torch.bfloat16):
        batch1 = torch.rand((1024, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_float16_5(self, dtype=torch.bfloat16):
        batch1 = torch.rand((1024, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_6(self, dtype=torch.float):
        batch1 = torch.rand((1024, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 4096), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_6(self, dtype=torch.bfloat16):
        batch1 = torch.rand((1024, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 4096), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )
    
    def test_mm_float16_6(self, dtype=torch.bfloat16):
        batch1 = torch.rand((1024, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 4096), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_7(self, dtype=torch.float):
        batch1 = torch.rand((4096, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_7(self, dtype=torch.bfloat16):
        batch1 = torch.rand((4096, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_float16_7(self, dtype=torch.bfloat16):
        batch1 = torch.rand((4096, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_8(self, dtype=torch.float):
        batch1 = torch.rand((30522, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_8(self, dtype=torch.bfloat16):
        batch1 = torch.rand((30522, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_float16_8(self, dtype=torch.bfloat16):
        batch1 = torch.rand((30522, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_9(self, dtype=torch.float):
        batch1 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_9(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_float16_9(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_10(self, dtype=torch.float):
        batch1 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 4096), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_10(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 4096), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_float16_10(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 1024), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1024, 4096), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_11(self, dtype=torch.float):
        batch1 = torch.rand((512, 4096), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((4096, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_mm_bfloat16_11(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 4096), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((4096, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_float16_11(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 4096), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((4096, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )

    def test_mm_12(self, dtype=torch.float):
        batch1 = torch.rand((512, 30522), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((30522, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-2, atol=1e-3
        )

    def test_mm_bfloat16_12(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 30522), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((30522, 1024), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-2, atol=1e-3
        )

    def test_mm_float16_12(self, dtype=torch.bfloat16):
        batch1 = torch.rand((512, 30522), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((30522, 1024), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.mm(batch1, batch2),
            torch.mm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-3
        )
