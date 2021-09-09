import unittest
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP, llga_test_env
from torch.testing._internal.common_utils import TEST_SCIPY

import intel_extension_for_pytorch as ipex

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

class TestOp(JitLlgaTestCase):
    @llga_test_env
    def test_linear(self):
        for freeze in [True, False]:
            for bias in [True, False]:
                x = torch.randn(32, 28)
                m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)

                graph, _ = self.checkTrace(m, [x], freeze)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
                self.assertFused(graph, ['aten::linear'])

    @llga_test_env
    def test_bmm(self):
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()

            def forward(self, x, y):
                return x.matmul(y)

        x = torch.randn(128, 16, 384, 64)
        y = torch.randn(128, 16, 64, 384)
        m = M()

        graph, _ = self.checkTrace(m, [x, y])
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::matmul'])

if __name__ == '__main__':
    run_tests()
