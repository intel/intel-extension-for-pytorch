import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_batch_linear_algebra(self, dtype=torch.float):
        x_cpu = torch.randn(5, 5)

        x_dpcpp = x_cpu.to(dpcpp_device)
        #y_cpu1 = x_cpu.new_ones((2, 3))
        y_cpu1 = torch.randn(5, 5)
        #y_cpu2 = x_cpu.new_ones((2, 3))
        y_cpu2 = torch.randn(5, 5)

        y_dpcpp1 = y_cpu1.to(dpcpp_device)
        y_dpcpp2 = y_cpu2.to(dpcpp_device)

        print("y_cpu", torch.tril(y_cpu2))
        print("y_dpcpp", torch.tril(y_dpcpp2).to("cpu"))
        self.assertEqual(torch.tril(y_cpu2),
                         torch.tril(y_dpcpp2).to(cpu_device))

        print("y_cpu", torch.triu(y_cpu2))
        print("y_dpcpp", torch.triu(y_dpcpp2).to("cpu"))
        self.assertEqual(torch.triu(y_cpu2),
                         torch.triu(y_dpcpp2).to(cpu_device))
