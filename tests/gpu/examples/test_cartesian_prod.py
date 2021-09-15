import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device('cpu')
xpu_device = torch.device('xpu')


class TestTorchMethod(TestCase):
    def test_cartesian_prod(self, dtype=torch.float):
        a_cpu = torch.tensor([1, 2, 3]).to(cpu_device)
        b_cpu = torch.tensor([4, 5]).to(cpu_device)
        y_cpu = torch.cartesian_prod(a_cpu, b_cpu)

        a_xpu = a_cpu.to(xpu_device)
        b_xpu = b_cpu.to(xpu_device)
        y_xpu = torch.cartesian_prod(a_xpu, b_xpu)

        self.assertEqual(y_cpu, y_xpu)
