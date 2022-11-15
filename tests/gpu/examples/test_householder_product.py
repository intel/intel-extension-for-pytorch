import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_householder_product(self, dtype=torch.float):
        h = torch.randn(3, 2, 2)
        tau = torch.randn(3, 1)
        cpu = torch.linalg.householder_product(h, tau)
        h_xpu = h.to("xpu")
        tau_xpu = tau.to("xpu")
        xpu = torch.linalg.householder_product(h_xpu, tau_xpu)
        self.assertEqual(cpu, xpu.cpu())

    def test_householder_product_out(self, dtype=torch.float):
        h = torch.randn(3, 2, 2)
        tau = torch.randn(3, 1)
        cpu = torch.empty_like(h)
        torch.linalg.householder_product(h, tau, out=cpu)
        h_xpu = h.to("xpu")
        tau_xpu = tau.to("xpu")
        xpu = torch.empty_like(h_xpu).to("xpu")
        torch.linalg.householder_product(h_xpu, tau_xpu, out=xpu)
        self.assertEqual(cpu, xpu.cpu())
