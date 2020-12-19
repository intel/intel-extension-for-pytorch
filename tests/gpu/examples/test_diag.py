import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_diag(self, dtype=torch.float):
        x_cpu1 = torch.randn(3, device=cpu_device)
        y1_cpu1 = torch.diag(x_cpu1)
        y2_cpu1 = torch.diag(x_cpu1, 1)
        print("cpu1 y1", y1_cpu1)
        print("cpu1 y2", y2_cpu1)

        x_dpcpp1 = x_cpu1.to(dpcpp_device)
        y1_dpcpp1 = torch.diag(x_dpcpp1)
        y2_dpcpp1 = torch.diag(x_dpcpp1, 1)
        print("syc1 y1", y1_dpcpp1.cpu())
        print("syc1 y2", y2_dpcpp1.cpu())
        self.assertEqual(x_cpu1, x_dpcpp1.to(cpu_device))
        self.assertEqual(y1_cpu1, y1_dpcpp1.to(cpu_device))
        self.assertEqual(y2_cpu1, y2_dpcpp1.to(cpu_device))

        x_cpu2 = torch.randn(3, 3, device=cpu_device)
        y1_cpu2 = torch.diag(x_cpu2)
        y2_cpu2 = torch.diag(x_cpu2, 1)
        print("cpu2 y1", y1_cpu2)
        print("cpu2 y2", y2_cpu2)

        x_dpcpp2 = x_cpu2.to(dpcpp_device)
        y1_dpcpp2 = x_dpcpp2.diag(0)
        y2_dpcpp2 = x_dpcpp2.diag(1)
        print("syc2 y1", y1_dpcpp2.cpu())
        print("syc2 y2", y2_dpcpp2.cpu())
        self.assertEqual(x_cpu2, x_dpcpp2.to(cpu_device))
        self.assertEqual(y1_cpu2, y1_dpcpp2.to(cpu_device))
        self.assertEqual(y2_cpu2, y2_dpcpp2.to(cpu_device))
