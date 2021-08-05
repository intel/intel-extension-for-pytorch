import unittest
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck

from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP, llga_test_env

import intel_pytorch_extension as ipex


class TestIpexOps(JitLlgaTestCase):
    @llga_test_env
    def test_adaptive_avg_pool2d(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((5,7))

            def forward(self, x):
                x = self.adaptive_avg_pool2d(x)
                return x

        m = M()
        x = torch.rand(1, 32, 28, 28)
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="adaptive_avg_pool2d", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)


    @llga_test_env
    def test_flatten_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, 2, padding=1, bias=True)
                self.pool = nn.MaxPool2d(2)
                self.flatten = nn.Flatten(1)
                self.linear = nn.Linear(147, 32)

            def forward(self, x):
                x = self.conv1(x)
                x = self.pool(x)
                x = self.flatten(x)
                x = self.linear(x)
                return x

        m = M()
        x = torch.rand(1, 3, 14, 14)
        patterns = [
            ["aten::quantize_per_channel", "aten::dequantize", "aten::_convolution"],
            ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
            ["aten::quantize_per_channel", "aten::dequantize", "aten::linear"],
        ]
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="flatten", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.checkPatterns(graph, patterns)

    @llga_test_env
    def test_flatten_fp32(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.flatten = nn.Flatten(1)

            def forward(self, x):
                x = self.flatten(x)
                return x

        m = M()
        x = torch.rand(1, 3, 14, 14)
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], config_name="flatten", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)
            FileCheck().check_not("aten::quantize_per_tensor") \
                .check_not("at::dequantize") \
                .check("aten::flatten") \
                .run(graph)



if __name__ == '__main__':
    run_tests()
