import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_scatter_add(self, dtype=torch.float):

        x1 = torch.rand(2, 10, device=cpu_device)
        x2 = torch.ones(3, 10, device=cpu_device)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        x2.scatter_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=cpu_device), x1)
        print(x2)

        x2_dpcpp.scatter_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=dpcpp_device), x1_dpcpp)
        print(x2_dpcpp.cpu())
        self.assertEqual(x2, x2_dpcpp.cpu())

        x1 = torch.rand(2, 10, device=cpu_device)
        x2 = torch.ones(3, 10, device=cpu_device)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        x2.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
                                        2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=cpu_device), x1)
        print(x2)

        x2_dpcpp.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=dpcpp_device), x1_dpcpp)
        print(x2_dpcpp.cpu())
        self.assertEqual(x2, x2_dpcpp.cpu())

        x1 = torch.tensor([[1, 2], [3, 4]],
                          dtype=torch.int32, device=cpu_device)
        x2 = torch.tensor([[1, 2], [3, 4], [5, 6]],
                          dtype=torch.int32, device=cpu_device)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        x2.scatter_add_(0, torch.tensor(
            [[0, 1], [2, 0]], device=cpu_device), x1)
        print(x2)

        x2_dpcpp.scatter_add_(0, torch.tensor(
            [[0, 1], [2, 0]], device=dpcpp_device), x1_dpcpp)
        print(x2_dpcpp.cpu())
        self.assertEqual(x2, x2_dpcpp.cpu())

        x3 = torch.rand(2, 10, device=cpu_device)
        x4 = torch.ones(3, 10, device=cpu_device)
        x3_dpcpp = x3.to("xpu")
        x4_dpcpp = x4.to("xpu")
        x4 = torch.zeros_like(x4).scatter_add(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=cpu_device), x3)
        print(x4)

        x4_dpcpp = torch.zeros_like(x4_dpcpp).scatter_add(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=dpcpp_device), x3_dpcpp)
        print(x4_dpcpp.cpu())
        self.assertEqual(x4, x4_dpcpp.cpu())

        x3 = torch.tensor([[1, 2], [3, 4]],
                          dtype=torch.int32, device=cpu_device)
        x4 = torch.tensor([[1, 2], [3, 4], [5, 6]],
                          dtype=torch.int32, device=cpu_device)
        x3_dpcpp = x3.to("xpu")
        x4_dpcpp = x4.to("xpu")
        x4 = torch.zeros_like(x4).scatter_add(0, torch.tensor(
            [[0, 1], [2, 0]], device=cpu_device), x3)
        print(x4)

        x4_dpcpp = torch.zeros_like(x4_dpcpp).scatter_add(0, torch.tensor(
            [[0, 1], [2, 0]], device=dpcpp_device), x3_dpcpp)
        print(x4_dpcpp.cpu())
        self.assertEqual(x4, x4_dpcpp.cpu())
