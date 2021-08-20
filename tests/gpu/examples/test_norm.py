import numpy
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_norm(self, dtype=torch.float):
        x_cpu = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float32)
        print(x_cpu.norm(p='fro', dim=[0, 1]))
        print(x_cpu.norm(p='fro', dim=[0]))

        x_dpcpp = torch.tensor(
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float32, device=dpcpp_device)
        print(x_dpcpp.norm(p='fro', dim=[0, 1]).cpu())
        print(x_dpcpp.norm(p='fro', dim=[0]).cpu())
        self.assertEqual(x_cpu.norm(p='fro', dim=[0, 1]), x_dpcpp.norm(
            p='fro', dim=[0, 1]).cpu())
        self.assertEqual(x_cpu.norm(
            p='fro', dim=[0]), x_dpcpp.norm(p='fro', dim=[0]).cpu())
