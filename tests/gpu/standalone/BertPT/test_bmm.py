import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_bmm_1(self, dtype=torch.float):
        batch1 = torch.randn((16, 512, 64), device=cpu_device)
        batch2 = torch.randn((16, 64, 512), device=cpu_device)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-5, atol=10e-5
        )
    
    def test_bmm_2(self, dtype=torch.float):
        batch1 = torch.rand((16, 512, 512), device=cpu_device)
        batch2 = torch.rand((16, 512, 64), device=cpu_device)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=1e-5, atol=1e-5
        )
    
    def test_bmm_3(self, dtype=torch.float):
        batch1 = torch.randn((16, 64, 512), device=cpu_device)
        batch2 = torch.randn((16, 512, 512), device=cpu_device)
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        self.assertEqual(
            torch.bmm(batch1, batch2),
            torch.bmm(batch1_dpcpp, batch2_dpcpp).to(cpu_device),
            rtol=10e-5, atol=10e-5
        )
