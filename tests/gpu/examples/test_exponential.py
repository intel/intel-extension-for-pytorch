import torch
#import matplotlib.pyplot as plt
import torch_ipex
from torch.testing._internal.common_utils import TestCase
import pytest

dpcpp_device = torch.device("xpu")
cpu_device = torch.device("cpu")


class  TestTorchMethod(TestCase):
    def test_exponential(self, dtype=torch.float):
        # Will not compare the results due to random seeds
        exp_cpu = torch.ones(1000000, device=cpu_device,dtype = dtype)
        exp_dist = exp_cpu.to("xpu")
        exp_cpu.exponential_(1)
        exp_dist.exponential_(1)
        # self.assertEqual(exp_cpu, exp_dist.cpu())

        print("exponential device ", exp_dist.device)
        print("exponential ", exp_dist.to("cpu"))

        # np_data = exp_dist.cpu().detach().numpy()

        # print("numpy ", np_data)
        # plt.hist(np_data, 100)
        # plt.show()


