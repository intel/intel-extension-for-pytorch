import torch
from torch.testing._internal.common_utils import TestCase

import torch_ipex
import pytest


torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class M(torch.nn.Module):
    def forward(self, a, b, c):
        o = a * b
        o += c
        return o


class TestJITMethod(TestCase):
    def test_fusion(self, dtype=torch.float):
        a = torch.randn((1, 2, 3, 3), dtype=torch.float, device=dpcpp_device)
        b = torch.randn((1, 2, 3, 3), dtype=torch.float, device=dpcpp_device)
        c = torch.randn((1, 2, 3, 3), dtype=torch.float, device=dpcpp_device)

        model = M().eval()
        ref = model(a, b, c).to("cpu")
        print("eager: ", ref)

        modelJit = torch.jit.script(model)
        with torch.no_grad():
            jit = modelJit(a, b, c).to("cpu")
            print("jit:", jit)

        self.assertEqual(ref, jit)
