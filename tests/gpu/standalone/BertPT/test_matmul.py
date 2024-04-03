import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
            ((1, 16, 512, 64), (1, 16, 64, 512)),
            ((1, 16, 512, 512), (1, 16, 512, 64)),
            ]

class TestTorchMethod(TestCase):
    def test_matmul_1(self, dtype=torch.float):
        batch1 = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 16, 64, 512), device=cpu_device, dtype=dtype)

        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.matmul(batch1, batch2),
            torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_matmul_bfloat16_1(self, dtype=torch.bfloat16):  
        batch1 = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 16, 64, 512), device=cpu_device, dtype=dtype)

        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.matmul(batch1, batch2),
            torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_matmul_float16_1(self, dtype=torch.bfloat16):
        batch1 = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 16, 64, 512), device=cpu_device, dtype=dtype)

        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.matmul(batch1, batch2),
            torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
        )

    def test_matmul_2(self, dtype=torch.float):
        batch1 = torch.rand((1, 16, 512, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((1, 16, 512, 64), device=cpu_device, dtype=dtype)

        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.matmul(batch1, batch2),
            torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_matmul_bfloat16_2(self, dtype=torch.bfloat16):  
        batch1 = torch.randn((1, 16, 512, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)

        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.matmul(batch1, batch2),
            torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
        )

    def test_matmul_float16_2(self, dtype=torch.bfloat16):
        batch1 = torch.randn((1, 16, 512, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((1, 16, 512, 64), device=cpu_device, dtype=dtype)

        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.matmul(batch1, batch2),
            torch.matmul(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
        )
