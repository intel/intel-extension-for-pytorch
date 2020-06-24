import torch
from torch.testing._internal.common_utils import TestCase
# import time
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_sort(self, dtype=torch.float):

        x_cpu = torch.randn(3, 4)
        sorted_cpu, indices_cpu = torch.sort(x_cpu)
        print("x_cpu = ", x_cpu, "sorted = ",
              sorted_cpu, "indices = ", indices_cpu)

        x_dpcpp = x_cpu.to("dpcpp")
        sorted_dpcpp, indices_dpcpp = torch.sort(x_dpcpp)
        print("x_dpcpp = ", x_dpcpp.cpu(), "sorted_dpcpp = ",
              sorted_dpcpp.cpu(), "indices_dpcpp", indices_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(sorted_cpu, sorted_dpcpp.cpu())
        self.assertEqual(indices_cpu, indices_dpcpp.cpu())
