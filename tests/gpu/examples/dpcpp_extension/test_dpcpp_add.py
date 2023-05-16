import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestDPCPPExtensionMethod(TestCase):
    @pytest.mark.skipif(True, reason="Building dpcp extension with ninja has a link error.")
    def test_add_ninja(self):
        import test_add_ninja
        a = torch.rand(2, 3).to(dpcpp_device)
        b = torch.rand(2, 3).to(dpcpp_device)
        c = torch.empty_like(a)
        d= a + b
        test_add_ninja.add(a, b, c)
        self.assertEqual(d.to(cpu_device), c.to(cpu_device))

    def test_add_non_ninja(self):
        import test_add_non_ninja
        a = torch.rand(2, 3).to(dpcpp_device)
        b = torch.rand(2, 3).to(dpcpp_device)
        c = torch.empty_like(a)
        d= a + b
        test_add_non_ninja.add(a, b, c)
        self.assertEqual(d.to(cpu_device), c.to(cpu_device))
