import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
@pytest.mark.skipif("not torch.xpu.has_onemkl()")
class TestTorchMethod(TestCase):
    def test_linalg_solve_2d(self, dtype=torch.float):
        A = torch.randn([3, 3], dtype=torch.complex64)
        b = torch.randn(3, dtype=torch.complex64)
        A_xpu = A.to(dpcpp_device)
        b_xpu = b.to(dpcpp_device)
        x = torch.linalg.solve(A, b)
        x_xpu = torch.linalg.solve(A_xpu, b_xpu)
        self.assertEqual(x, x_xpu.cpu())

    def test_linalg_solve_3d(self, dtype=torch.float):
        A = torch.randn([2, 3, 3], dtype=torch.complex64)
        b = torch.randn([2, 3, 4], dtype=torch.complex64)
        A_xpu = A.to(dpcpp_device)
        b_xpu = b.to(dpcpp_device)
        x = torch.linalg.solve(A, b)
        x_xpu = torch.linalg.solve(A_xpu, b_xpu)
        self.assertEqual(x, x_xpu.cpu())
