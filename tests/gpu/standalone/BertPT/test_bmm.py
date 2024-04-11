import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_bmm_1(self, dtype=torch.float):
        batch1 = torch.randn((16, 512, 64), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-5, atol=10e-5
        )

    def test_bmm_bfloat16_1(self, dtype=torch.bfloat16):
        batch1 = torch.randn((16, 512, 64), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-3, atol=10e-3
        )

    def test_bmm_float16_1(self, dtype=torch.bfloat16):
        batch1 = torch.randn((16, 512, 64), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=10e-3, atol=10e-3
        )
    
    def test_bmm_2(self, dtype=torch.float):
        batch1 = torch.rand((16, 512, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((16, 512, 64), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_bmm_bfloat16_2(self, dtype=torch.bfloat16):
        batch1 = torch.rand((16, 512, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((16, 512, 64), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )

    def test_bmm_float16_2(self, dtype=torch.bfloat16):
        batch1 = torch.rand((16, 512, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.rand((16, 512, 64), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=1e-2, atol=1e-2
        )
    
    def test_bmm_3(self, dtype=torch.float):
        batch1 = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((16, 512, 512), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-5, atol=10e-5
        )

    def test_bmm_bfloat16_3(self, dtype=torch.bfloat16):
        batch1 = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((16, 512, 512), device=cpu_device, dtype=dtype)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-3, atol=10e-3
        )

    def test_bmm_float16_3(self, dtype=torch.bfloat16):
        batch1 = torch.randn((16, 64, 512), device=cpu_device, dtype=dtype)
        batch2 = torch.randn((16, 512, 512), device=cpu_device, dtype=dtype)
        dtype_dpcpp = torch.float16
        batch1_dpcpp = batch1.to(dpcpp_device).to(dtype_dpcpp)
        batch2_dpcpp = batch2.to(dpcpp_device).to(dtype_dpcpp)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device).to(torch.bfloat16),
            rtol=10e-3, atol=10e-3
        )
