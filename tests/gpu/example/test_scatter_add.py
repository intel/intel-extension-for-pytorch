import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_scatter_add(self, dtype=torch.float):

        x1 = torch.rand(2, 10, device=cpu_device)
        x2 = torch.ones(3, 10, device=cpu_device)
        x1_dpcpp = x1.to("dpcpp")
        x2_dpcpp = x2.to("dpcpp")
        x2.scatter_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=cpu_device), x1)
        print(x2)

        x2_dpcpp.scatter_(0, torch.tensor([[0, 1, 2, 0, 0, 0, 1, 2, 0, 0], [
            2, 0, 0, 1, 2, 2, 0, 0, 1, 2]], device=dpcpp_device), x1_dpcpp)
        print(x2_dpcpp.cpu())
        self.assertEqual(x2, x2_dpcpp.cpu())

        x1 = torch.rand(2, 10, device=cpu_device)
        x2 = torch.ones(3, 10, device=cpu_device)
        x1_dpcpp = x1.to("dpcpp")
        x2_dpcpp = x2.to("dpcpp")
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
        x1_dpcpp = x1.to("dpcpp")
        x2_dpcpp = x2.to("dpcpp")
        x2.scatter_add_(0, torch.tensor(
            [[0, 1], [2, 0]], device=cpu_device), x1)
        print(x2)

        x2_dpcpp.scatter_add_(0, torch.tensor(
            [[0, 1], [2, 0]], device=dpcpp_device), x1_dpcpp)
        print(x2_dpcpp.cpu())
        self.assertEqual(x2, x2_dpcpp.cpu())
