import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_atan2(self, dtype=torch.float):
        x_cpu = torch.randn(4, device=cpu_device)
        y_cpu = torch.randn(4, device=cpu_device)

        dist_cpu = torch.atan2(x_cpu, y_cpu)

        print("torch.atan2(x_cpu, y_cpu)", dist_cpu)

        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = y_cpu.to(dpcpp_device)

        dist_dpcpp = torch.atan2(x_dpcpp, y_dpcpp)

        print("torch.atan2(x_dpcpp, y_dpcpp)", dist_dpcpp.cpu())

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(dist_cpu, dist_dpcpp.to(cpu_device))
