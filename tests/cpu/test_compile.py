import unittest
import itertools
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import intel_extension_for_pytorch as ipex

from common_utils import TestCase


class Conv_Bn_Relu(nn.Module):
    def __init__(self):
        super(Conv_Bn_Relu, self).__init__()

        self.conv = nn.Conv2d(6, 3, 3)
        self.bn = nn.BatchNorm2d(3, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class TestCompile(TestCase):
    def test_inference(self):
        model_ = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
        x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)
        for dtype, ipex_optimize in itertools.product(
            [torch.float32, torch.bfloat16], [True, False]
        ):
            model = copy.deepcopy(model_)
            if ipex_optimize:
                model = ipex.optimize(model, dtype=dtype)
            with torch.cpu.amp.autocast(
                enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16
            ), torch.no_grad():
                y1 = model(x)
                fx_model = torch.fx.symbolic_trace(model)
                compiled_model = ipex.compile(fx_model, [x])
                # warm up
                for _ in range(2):
                    compiled_model(x)
                y2 = compiled_model(x)
            self.assertEqual(y1, y2)
            self.assertTrue(y2.dtype == dtype)


if __name__ == "__main__":
    test = unittest.main()
