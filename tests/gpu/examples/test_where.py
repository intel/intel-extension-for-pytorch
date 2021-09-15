import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_where(self, dtype=torch.float):
        x = torch.tensor([[0.6580, -1.0969, -0.4614],
                          [-0.1034, -0.5790, 0.1497]])
        x_ones = torch.tensor([[1., 1., 1.], [1., 1., 1.]])
        print("cpu", torch.where(x > 0, x, x_ones))

        x_dpcpp = torch.tensor([[0.6580, -1.0969, -0.4614],
                                [-0.1034, -0.5790, 0.1497]], device=torch.device("xpu"))
        x_ones_dpcpp = torch.tensor(
            [[1., 1., 1.], [1., 1., 1.]], device=torch.device("xpu"))
        print("xpu", torch.where(x_dpcpp > 0, x_dpcpp, x_ones_dpcpp).cpu())
        self.assertEqual(x, x_dpcpp.to(cpu_device))
        self.assertEqual(x_ones, x_ones_dpcpp.to(cpu_device))
        self.assertEqual(torch.where(x > 0, x, x_ones), torch.where(
            x_dpcpp > 0, x_dpcpp, x_ones_dpcpp).cpu())
