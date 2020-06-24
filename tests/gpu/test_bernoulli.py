import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    @pytest.mark.skip(reason='Random Data Generate')
    def test_bernoulli(self, dtype=torch.float):
        user_cpu = torch.empty(3, 3).uniform_(0, 1)
        cpu_res = torch.bernoulli(user_cpu, 0.5)
        print("Raw Array")
        print(user_cpu)
        print("CPU Res")
        print(cpu_res)
        dpcpp_res = torch.bernoulli(user_cpu.to(dpcpp_device))
        print("dpcpp Res")
        print(dpcpp_res.cpu())
        dpcpp_res = torch.bernoulli(user_cpu.to(dpcpp_device), 0.5)
        print("dpcpp Res")
        print(dpcpp_res.cpu())
        self.assertEqual(cpu_res, dpcpp_res.to(cpu_device))
