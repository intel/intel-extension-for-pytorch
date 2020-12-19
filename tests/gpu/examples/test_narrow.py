import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_narrow(self, dtype=torch.float):

        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x.narrow(0, 0, 2)
        x_dpcpp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                               device=torch.device("xpu"))
        x_dpcpp.narrow(0, 0, 2)
        print("x = ", x)
        print("x_dpcpp = ", x_dpcpp.to("cpu"))
        self.assertEqual(x, x_dpcpp.cpu())
