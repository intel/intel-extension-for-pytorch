import torch
import ipex
import numpy as np
import random
from torch.testing._internal.common_utils import TestCase


class TestNNMethod(TestCase):
    def test_sort(self, dtype=torch.float):
        for i in range(100):
            a = random.randint(1, 3)
            b = random.randint(1, 5)
            c = random.randint(1, 7)
            d = random.randint(1024, 5000)
            abcd = [a, b, c, d]
            random.shuffle(abcd)
            x_cpu = torch.randn(abcd).to(dtype)

            x_cpu = torch.testing.make_non_contiguous(x_cpu)
            # print(x_cpu.stride())

            x_xpu = x_cpu.clone().xpu()
            x_xpu = torch.testing.make_non_contiguous(x_xpu)
            dim = random.randint(0, 3)
            # print(abcd, dim)

            x_cpu = torch.sort(x_cpu, dim=dim)[0]
            prof = False
            with torch.autograd.profiler.profile(prof, use_xpu=True) as prof:
                x_xpu = torch.sort(x_xpu, dim=dim)[0]
                # print(x_xpu.cpu())
            if prof:
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                print(prof.table(sort_by="id", row_limit=100000))
            maxdiff = float((x_cpu - x_xpu.cpu().float()).abs().max())
            print(abcd, ', dim:', dim, ', maxdiff:', maxdiff)
            assert(maxdiff < 1e-5)
