import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_bincount(self, dtype=torch.float):
        x_cpu = torch.randint(0, 8, (5,), dtype=torch.int64)
        y_cpu = torch.linspace(0, 1, steps=5)
        x_dpcpp = x_cpu.to(dpcpp_device)
        y_dpcpp = y_cpu.to(dpcpp_device)

        print("bincount cpu 1", torch.bincount(x_cpu))
        # print("bincount cpu 2" x_cpu.bincount(y_cpu))
        print("bincount dpcpp 1", torch.bincount(x_dpcpp).cpu())
        # print("bincount dpcpp 2" x_dpcpp.bincount(y_dpcpp).cpu())
        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(torch.bincount(x_cpu),
                         torch.bincount(x_dpcpp).to(cpu_device))
