# import matplotlib.pyplot as plt
import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

dpcpp_device = torch.device("xpu")
cpu_device = torch.device("cpu")


class TestTorchMethod(TestCase):
    def test_exponential(self, dtype=torch.float):
        #  Will not compare the results due to random seeds
        exp = torch.ones(1000000, device=dpcpp_device, dtype=dtype)
        exp_1 = torch.ones(1000000, device=dpcpp_device, dtype=dtype)
        exp_dist = exp
        exp_dist_1 = exp_1
        torch.xpu.manual_seed(100)
        exp_dist.exponential_(1)
        torch.xpu.manual_seed(100)
        exp_dist_1.exponential_(1)
        print("exp_dist device:", exp_dist.device)
        print("exp_dist_1 device:", exp_dist_1.device)
        self.assertEqual(exp_dist.cpu(), exp_dist_1.cpu())
