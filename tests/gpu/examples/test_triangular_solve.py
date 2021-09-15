import time

import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not ipex._onemkl_is_enabled()")
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
