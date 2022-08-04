import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_cross(self, dtype=torch.float):

        a = torch.randn((4, 3), device=cpu_device)
        b = torch.randn((4, 3), device=cpu_device)

        print(a.cross(b))

        a_dpcpp = a.to(dpcpp_device)
        b_dpcpp = b.to(dpcpp_device)
        print(a_dpcpp.cross(b_dpcpp).cpu())

        self.assertEqual(a, a_dpcpp.to(cpu_device))
        self.assertEqual(b, b_dpcpp.to(cpu_device))
        self.assertEqual(a.cross(b), a_dpcpp.cross(b_dpcpp).to(cpu_device))
