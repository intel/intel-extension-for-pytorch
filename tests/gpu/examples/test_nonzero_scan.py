import torch
from torch.testing._internal.common_utils import TestCase
import ipex

torch.xpu.manual_seed(0)


class TestTorchMethod(TestCase):
    def test_nonzero_scan(self):
        for i in range(1, 1000):
            a = torch.rand(4, 15000)
            b = torch.rand(4, 15000)
            d = a.to('xpu')
            e = b.to('xpu')
            f = d < e
            dd = d[f]
            ee = e[f]
            self.assertTrue(dd.size(0) == ee.size(0))
