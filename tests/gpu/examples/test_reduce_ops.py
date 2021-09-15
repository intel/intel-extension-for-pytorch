import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import numpy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):

    def test_reduce_ops(self, dtype=torch.float):
        user_cpu = torch.randn([256, 3, 2, 4], device=cpu_device)
        print(user_cpu)
        res_cpu = torch.sum(user_cpu, 0, False)
        print("cpu result:")
        print(res_cpu)
        print("begin dpcpp compute:")
        res_dpcpp = torch.sum(user_cpu.to("xpu"), 0, False)
        print("dpcpp result:")
        print(res_dpcpp.cpu())
        self.assertEqual(res_cpu, res_dpcpp.cpu())
