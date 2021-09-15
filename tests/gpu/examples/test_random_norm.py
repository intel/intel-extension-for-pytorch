import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_random_norm(self, dtype=torch.float):
        x_cpu = torch.tensor([1.111, 2.222, 3.333, 4.444, 5.555, 6.666], device=cpu_device, dtype=dtype)
        x_dpcpp = torch.tensor([1.111, 2.222, 3.333, 4.444, 5.555, 6.666], device=dpcpp_device, dtype=dtype)

        print("normal_ cpu", x_cpu.normal_(2.0, 0.5))
        print("normal_ dpcpp", x_dpcpp.normal_(2.0, 0.5).cpu())

        self.assertEqual(x_dpcpp.mean(), 2.0, rtol=0.3, atol=0.3)
        self.assertEqual(x_dpcpp.std(), 0.5, rtol=0.3, atol=0.3)
