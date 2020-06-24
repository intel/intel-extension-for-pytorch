import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_median(self, dtype=torch.float):
        x_cpu = torch.randn(2, 3)
        x_dpcpp = x_cpu.to("dpcpp")

        print("x_cpu", x_cpu, " median_cpu", torch.median(x_cpu))
        print("x_dpcpp", x_dpcpp.to("cpu"), " median_dpcpp",
              torch.median(x_dpcpp).to("cpu"))
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(torch.median(x_cpu), torch.median(x_dpcpp).to("cpu"))

        x_cpu2 = torch.tensor(([1, 2, 3, 4, 5]), dtype=torch.int32,
                              device=torch.device("cpu"))
        x_dpcpp2 = torch.tensor(
            ([1, 2, 3, 4, 5]), dtype=torch.int32, device=torch.device("dpcpp"))

        print("x_cpu2", x_cpu2, " median_cpu2", x_cpu2.median())
        print("x_dpcpp2", x_dpcpp2.to("cpu"),
              " median_dpcpp2", x_dpcpp2.median().to("cpu"))
        self.assertEqual(x_cpu2, x_dpcpp2.cpu())
        self.assertEqual(torch.median(x_cpu2),
                         torch.median(x_dpcpp2).to("cpu"))
