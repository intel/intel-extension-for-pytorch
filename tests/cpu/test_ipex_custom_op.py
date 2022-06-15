import torch
import intel_extension_for_pytorch as ipex
import unittest
from common_utils import TestCase

class TestCustomOp(TestCase):
    # Port from test_torch
    def test_add_softmax(self):
        # smaller input which can't can in AVX512
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        orig_result = a.add(b).softmax(-1)
        ipex_result = torch.ops.torch_ipex.add_softmax_(a, b) 
        self.assertEqual(orig_result, ipex_result)
        
        # bigger input which can in AVX512
        a = torch.randn(30, 30)
        b = torch.randn(30, 30)
        orig_result = a.add(b).softmax(-1)
        ipex_result = torch.ops.torch_ipex.add_softmax_(a, b) 
        self.assertEqual(orig_result, ipex_result)
        # broadcast
        a = torch.randn(30, 30)
        b = torch.randn(30)
        orig_result = a.add(b).softmax(-1)
        ipex_result = torch.ops.torch_ipex.add_softmax_(a, b) 
        self.assertEqual(orig_result, ipex_result)

if __name__ == '__main__':
    test = unittest.main()
