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
from torch.testing._internal.common_utils import TEST_SCIPY, TemporaryFileName

import intel_extension_for_pytorch as ipex
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, QConfig


class TestDefaultRecipe(JitLlgaTestCase):
    def test_quantized_op_int8_int8(self):
        # test some op only suport INT8+INT8, if its
        # post doesn't have a quantized op, we need make sure
        # it can also call in INT8 kernel.
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


if __name__ == '__main__':
    run_tests() 
