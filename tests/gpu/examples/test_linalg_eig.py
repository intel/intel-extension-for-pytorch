import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
class TestTorchMethod(TestCase):
    def test_linalg_eig(self, dtype=torch.complex128):
        input = torch.randn(2, 2, dtype=torch.complex128)
        input_xpu = input.to(dpcpp_device)

        L, V = torch.linalg.eig(input)
        L_xpu, V_xpu = torch.linalg.eig(input_xpu)
        self.assertEqual(L, L_xpu.to(cpu_device))
        self.assertEqual(V, V_xpu.to(cpu_device))
