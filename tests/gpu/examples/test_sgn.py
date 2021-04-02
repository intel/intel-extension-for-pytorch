import torch
import torch_ipex
import pytest

from torch.testing._internal.common_utils import TestCase

class  TestTorchMethod(TestCase):
    def test_sgn_float(self, dtype=torch.float):
        x = torch.randn(1,4)
        y = x.sgn()
        
        x_gpu = x.to('xpu')
        y_xpu = x_gpu.sgn()
        
        #print(y)
        #print(y_xpu.cpu())

        self.assertEqual(y, y_xpu.cpu())

    def test_sgn_double(self, dtype=torch.double):
        x = torch.randn(1,4).double()
        y = x.sgn()

        x_gpu = x.to('xpu')
        y_xpu = x_gpu.sgn()

        #print(y)
        #print(y_xpu.cpu())

        self.assertEqual(y, y_xpu.cpu())
