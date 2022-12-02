import sys
import os
import unittest
import itertools
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
import copy
from test_autocast import get_rand_seed

import intel_extension_for_pytorch as ipex
from test_ao_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP

import intel_extension_for_pytorch as ipex

class TestDefaultRecipe(JitLlgaTestCase):
    def test_quantized_op_int8_int8(self):
        # Test one op which only support INT8+INT8, if its
        # post op is not a quantifiable op, we need to make sure
        # it can also call in INT8 kernel by inset fake quant after it's output.
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.pool = nn.MaxPool2d(1, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                return x

        m = M()
        x = torch.rand(1, 2, 14, 14)
       
        graph = self.checkQuantizeTrace(m, [x], atol=2e-1)
        patterns = [
                ["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::quantize_per_tensor"],
                ["aten::dequantize", "aten::max_pool2d", "aten::quantize_per_tensor"],
            ]
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
        self.checkPatterns(graph, patterns)

    def test_none_gemm_op_has_quantized_op_before(self):
        # For none-gemm op, if it's pre op is quantifiable op, fake quant will be inserted.
        # Given the following example, the quantization flow will be like:
        # q->dq->quantized_module->q->dq->flatten->q->dq.
        class M(nn.Module):
            def __init__(self, quantized_module):
                super(M, self).__init__()
                self.quantized_module = quantized_module

            def forward(self, x):
                x = self.quantized_module(x)
                x = x.flatten(1)
                return x

        class conv_swish(nn.Module):
            def __init__(self, ):
                super(conv_swish, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)

            def forward(self, x):
                x = self.conv(x)
                y = x.sigmoid()
                z = torch.mul(x, y)
                return z

        class conv_eltwise(nn.Module):
            def __init__(self, ):
                super(conv_eltwise, self).__init__()
                self.conv = torch.nn.Conv2d(2, 2, 1)

            def forward(self, x):
                x = self.conv(x)
                x = x.relu_()
                return x

        # TODO: test more quantized modules(especially for fused module). 
        quantized_modules = [conv_swish(), conv_eltwise()]
        patterns = [
                [["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::sigmoid", "aten::mul", "aten::quantize_per_tensor"]],
                [["aten::dequantize", "aten::dequantize", "aten::_convolution", "aten::relu", "aten::quantize_per_tensor"]],
            ]
        for quantized_modules, pattern in zip(quantized_modules, patterns):
            m = M(quantized_modules).eval()

            x = torch.rand(1, 2, 14, 14)

            graph = self.checkQuantizeTrace(m, [x], atol=2e-1)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.checkPatterns(graph, pattern)
            FileCheck().check("aten::dequantize").run(graph)

if __name__ == '__main__':
    run_tests() 
