import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest


class TestNNMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="not torch.xpu.has_onemkl()")
    def test_pinverse(self):
        a = torch.randn(5, 3)
        b = torch.pinverse(a)
        print('output cpu = ', b)

        a_xpu = a.detach().to('xpu')
        b_xpu = torch.pinverse(a_xpu)
        print('output xpu = ', b_xpu.cpu())

        self.assertEqual(b, b_xpu.cpu())

        a = torch.randn(3, 5)
        b = torch.pinverse(a)
        print('output cpu = ', b)

        a_xpu = a.detach().to('xpu')
        b_xpu = torch.pinverse(a_xpu)
        print('output xpu = ', b_xpu.cpu())

        self.assertEqual(b, b_xpu.cpu())
