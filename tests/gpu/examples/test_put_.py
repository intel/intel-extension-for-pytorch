import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_put(self, dtype=torch.float):

        x_cpu = torch.tensor([[4, 3, 5],
                              [6, 7, 8]], dtype=torch.int32)
        x_dpcpp = x_cpu.to(dpcpp_device)

        x_cpu.put_(torch.tensor([1, 3]), torch.tensor(
            [9, 10], dtype=torch.int32))

        print("x_cpu", x_cpu)

        x_dpcpp.put_(torch.tensor([1, 3]), torch.tensor(
            [9, 10], dtype=torch.int32))

        print("x_dpcpp", x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())

        x_cpu.put_(torch.tensor([1, 1]), torch.tensor(
            [9, 10], dtype=torch.int32), accumulate=True)

        print("x_cpu", x_cpu)

        x_dpcpp.put_(torch.tensor([1, 1]), torch.tensor(
            [9, 10], dtype=torch.int32), accumulate=True)

        print("x_dpcpp", x_dpcpp.cpu())
        self.assertEqual(x_cpu, x_dpcpp.cpu())
