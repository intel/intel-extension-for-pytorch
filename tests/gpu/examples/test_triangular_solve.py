
import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_triangular_solve(self, dtype=torch.float):
        a = torch.randn(2, 2)
        b = torch.randn(2, 3)
        c, d = torch.triangular_solve(b, a)

        # print(c)
        # print(d)

        a_xpu = a.to('xpu')
        b_xpu = b.to('xpu')
        c_xpu, d_xpu = torch.triangular_solve(b_xpu, a_xpu)

        # print(c_xpu.cpu())
        # print(d_xpu.cpu())

        self.assertEqual(c, c_xpu.cpu())
        self.assertEqual(d, d_xpu.cpu())

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_triangular_solve_double(self, dtype=torch.double):
        a = torch.randn([3, 3, 3]).double()
        b = torch.randn([3, 3, 3]).double()

        a_xpu = a.clone().xpu()
        b_xpu = b.clone().xpu()

        c, d = torch.triangular_solve(a, b)

        c_xpu, d_xpu = torch.triangular_solve(a_xpu, b_xpu)

        self.assertEqual(c, c_xpu.cpu())
        self.assertEqual(d, d_xpu.cpu())
