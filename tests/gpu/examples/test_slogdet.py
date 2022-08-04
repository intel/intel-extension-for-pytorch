
import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
    def test_slogdet(self, dtype=torch.float):
        A = torch.randn(3, 3)
        A_xpu = A.to("xpu")
        s, logdet = torch.slogdet(A)
        s_xpu, logdet_xpu = torch.slogdet(A_xpu)
        self.assertEqual(s, s_xpu.cpu())
        self.assertEqual(logdet, logdet_xpu.cpu())
