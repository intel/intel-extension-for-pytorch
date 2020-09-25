import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_eq(self, dtype=torch.float):
        x_cpu1 = torch.tensor([[1., 2.], [3., 4.]])
        x_cpu2 = torch.tensor([[1., 1.], [4., 4.]])
        x_cpu3 = torch.tensor([[1., 2.], [3., 4.]])
        y_cpu = torch.eq(x_cpu1, x_cpu2)
        print("eq cpu", y_cpu)
        print("equal cpu1", torch.equal(x_cpu1, x_cpu2))
        print("equal cpu2", torch.equal(x_cpu1, x_cpu3))
        self.assertEqual(False, torch.equal(x_cpu1, x_cpu2))
        self.assertEqual(True,  torch.equal(x_cpu1, x_cpu3))

        x_dpcpp1 = x_cpu1.to(dpcpp_device)
        x_dpcpp2 = x_cpu2.to(dpcpp_device)
        x_dpcpp3 = x_cpu3.to(dpcpp_device)

        y_dpcpp = torch.eq(x_dpcpp1, x_dpcpp2)
        print("eq dpcpp", y_dpcpp.cpu())
        print("eqeual dpcpp1", torch.equal(x_dpcpp1, x_dpcpp2))
        print("eqeual dpcpp2", torch.equal(x_dpcpp1, x_dpcpp3))
        self.assertEqual(False,  torch.equal(x_dpcpp1, x_dpcpp2))
        self.assertEqual(True,  torch.equal(x_dpcpp1, x_dpcpp3))
