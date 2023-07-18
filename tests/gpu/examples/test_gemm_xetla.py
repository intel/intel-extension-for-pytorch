import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


def gemm_xetla(m, n, k, dtype):
    input = torch.randn(m, k).type(dtype).xpu()
    input_cpu = input.cpu().float()
    linear = nn.Linear(k, n, bias=False).type(dtype).xpu()
    linear_cpu = nn.Linear(k, n, bias=False)
    linear_cpu.weight.requires_grad = False
    linear_cpu.weight[:] = linear.weight
    with torch.autograd.profiler_legacy.profile(use_xpu=True) as p:
        c = linear(input)
    # p.export_chrome_trace('test.json')
    print(p.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
    c_cpu = linear_cpu(input_cpu)
    maxdiff = (c.cpu() - c_cpu).abs().max().item()
    print(maxdiff)
    return c.cpu(), c_cpu


class TestNNMethod(TestCase):
    def test_gemm_xetla(self):
        shapes = [
            # m, n, k
            [1, 4096, 4096],
            [1, 4096, 16384],
            [1, 16384, 4096],
            [1, 32000, 4096]
        ]
        for shape in shapes:
            print(shape)
            out, ref = gemm_xetla(shape[0], shape[1], shape[2], torch.half)
            self.assertEqual(out.float(), ref, atol=1e-2, rtol=1e-2)
