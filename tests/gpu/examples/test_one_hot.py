import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_one_hot(self, dtype=torch.float):
        x_cpu = torch.arange(0, 5) % 3
        x_dpcpp = x_cpu.to("xpu")
        y_cpu = F.one_hot(x_cpu, num_classes=5)
        print(y_cpu)
        y_dpcpp = F.one_hot(x_dpcpp, num_classes=5)
        print(y_dpcpp.to("cpu"))
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
