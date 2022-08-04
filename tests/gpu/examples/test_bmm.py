import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_bmm(self, dtype=torch.float):
        M = torch.randn(3, 3, 5)
        batch1 = torch.randn((3, 3, 4), device=cpu_device)
        batch2 = torch.randn((3, 4, 5), device=cpu_device)

        #
        # Test bmm OP.
        #
        batch1_dpcpp = batch1.to(dpcpp_device)
        batch2_dpcpp = batch2.to(dpcpp_device)
        print("torch.bmm cpu", torch.bmm(batch1, batch2))
        print("torch.bmm dpcpp", torch.bmm(
            batch1_dpcpp, batch2_dpcpp).to('cpu'))
        print("tensor.bmm dpcpp", batch1_dpcpp.bmm(batch2_dpcpp).to('cpu'))
        self.assertEqual(torch.bmm(batch1, batch2), torch.bmm(
            batch1_dpcpp, batch2_dpcpp).to(cpu_device))
        #
        # Test bmm OP.
        #

        M_dpcpp = M.to("xpu")
        print("torch.baddbmm cpu", torch.baddbmm(M, batch1, batch2))
        print("torch.baddbmm dpcpp", torch.baddbmm(
            M_dpcpp, batch1_dpcpp, batch2_dpcpp).to('cpu'))
        print("tensor.baddbmm dpcpp", M_dpcpp.baddbmm(
            batch1_dpcpp, batch2_dpcpp).to('cpu'))
        self.assertEqual(torch.baddbmm(M, batch1, batch2), torch.baddbmm(
            M_dpcpp, batch1_dpcpp, batch2_dpcpp).to(cpu_device))
