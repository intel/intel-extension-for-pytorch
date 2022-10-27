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

    def test_inference_mode(self):
        class DemoModel(torch.nn.Module):
            def __init__(self):
                super(DemoModel, self).__init__()
                self.conv = torch.nn.Conv2d(3, 64, (3, 3))
                self.bn =  ipex.nn.FrozenBatchNorm2d(64)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        x = torch.rand((1, 3, 640, 640))
        model = DemoModel().eval()
        # enable weight prepack op.
        model = ipex.optimize(model)
        with torch.no_grad():
            y_ref = model(x)
        with torch.inference_mode():
            y_inf = model(x)
        self.assertEqual(y_ref, y_inf)


if __name__ == '__main__':
    test = unittest.main()
