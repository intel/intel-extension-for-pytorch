from functools import wraps

import torch
from torch.utils import ThroughputBenchmark
from torch.testing import assert_allclose
from torch.testing._internal.common_utils import run_tests, TestCase

import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core
from test_ao_jit_llga_utils import JitLlgaTestCase, run_tests, LLGA_FUSION_GROUP


class LinearEltwise(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(LinearEltwise, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.eltwise = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.eltwise(x)
        x = self.linear2(x)
        return x

def freeze(model):
    return torch.jit._recursive.wrap_cpp_module(torch._C._freeze_module(model._c, preserveParameters=True))

class TestThroughputBenchmark(JitLlgaTestCase):
    def test_linear_eltwise(self):
        with torch.no_grad():
            D_in = 10
            H = 5
            D_out = 15
            B = 8

            m = LinearEltwise(D_in, H, D_out)
            x = torch.randn(B, D_in)

            graph, m_llga, m_cpu = self.prepareModel(m, [x])

            ipex.enable_onednn_fusion(False)
            module_result = m_cpu(x)
            ipex.enable_onednn_fusion(True)

            bench = ThroughputBenchmark(m_llga)
            bench.add_input(x)
            bench_result = bench.run_once(x)

            assert_allclose(bench_result, module_result, atol=1e-1, rtol=1e-2)

            stats = bench.benchmark(
                num_calling_threads=4,
                num_warmup_iters=100,
                num_iters=1000
            )

            print(stats)

if __name__ == '__main__':
    run_tests()
