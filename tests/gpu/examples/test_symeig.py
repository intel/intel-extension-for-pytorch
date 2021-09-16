import copy
import time

import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_symeig(self, dtype=torch.float):
        a = torch.randn(5, 5)
        a = a + a.t()
        a_xpu = a.to("xpu")
        e, v = torch.symeig(a, eigenvectors=True)

        e_xpu, v_xpu = torch.symeig(a_xpu, eigenvectors=True)

        self.assertEqual(e, e_xpu.cpu())
        self.assertEqual(v, v_xpu.cpu())
