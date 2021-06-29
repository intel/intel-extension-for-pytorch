import torch
from torch.nn import functional as F
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest

class TestNNMethod(TestCase):
    @pytest.mark.skipif("not torch_ipex._onemkl_is_enabled()")
    def test_pinverse(self): 
        a = torch.randn(3, 5)
        b = torch.pinverse(a)
        print('output cpu = ', b)

        a_xpu = a.detach().to('xpu')
        b_xpu = torch.pinverse(a_xpu)
        print('output xpu = ', b_xpu.cpu())

        self.assertEqual(b, b_xpu.cpu(), rtol=1e-3, atol=1)