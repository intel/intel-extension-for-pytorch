import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_scatter(self, dtype=torch.float):

        x_cpu = torch.rand(2, 5)
        x_dpcpp = x_cpu.to(dpcpp_device)

        y_cpu = torch.zeros(3, 5).scatter_(0, torch.tensor(
            [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x_cpu)

        print("y_cpu", y_cpu)

        y_dpcpp = torch.zeros(3, 5, device=dpcpp_device).scatter_(0, torch.tensor(
            [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=dpcpp_device), x_dpcpp)

        print("y_dpcpp", y_dpcpp.cpu())

        z_cpu = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)

        print("z_cpu", z_cpu)

        z_dpcpp = torch.zeros(2, 4, device=dpcpp_device).scatter_(
            1, torch.tensor([[2], [3]], device=dpcpp_device), 1.23)

        print("z_dpcpp", z_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(z_cpu, z_dpcpp.cpu())
