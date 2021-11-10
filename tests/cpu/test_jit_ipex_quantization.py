import unittest
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck

from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP

import intel_extension_for_pytorch as ipex


class TestIpexOps(JitLlgaTestCase):
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
            ["aten::dequantize", "aten::_convolution"],
            ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
            ["aten::dequantize", "aten::linear"],
        ]
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [x], atol=2e-1, config_name="flatten", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)
            self.checkPatterns(graph, patterns)

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

    def test_embeddingbag_int8(self):
        m = nn.EmbeddingBag(10, 3, mode='sum', sparse=True)
        input = torch.LongTensor([1,2,4,5,4,3,2,9])
        offsets = torch.LongTensor([0,1,2,3,4,5,6,7])
        for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, [input, offsets], atol=1e-2, config_name="emb", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, 'ipex::qembedding_bag', 1)

    def test_interaction_int8(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.f = ipex.nn.functional.interaction

            def forward(self, *x):
                x = self.f(*x)
                return x

        m = M()
        inputs = []
        for i in range(0, 27):
            inputs.append(torch.randn([128, 128]) * 0.1)
        for qscheme in [torch.per_tensor_symmetric]:
            graph = self.checkQuantizeTrace(m, inputs, atol=1e-2, config_name="interaction", qscheme=qscheme)
            self.assertGraphContainsExactly(graph, 'ipex::qinteraction', 1)

if __name__ == '__main__':
    run_tests()
