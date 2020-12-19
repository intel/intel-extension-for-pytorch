import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_fill(self, dtype=torch.float):

        x = torch.ones([5, 3], device=cpu_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_fill_(0, index, -2)
        print("x = ", x)

        x_dpcpp = torch.ones([5, 3], device=dpcpp_device)
        index = torch.tensor([0, 4, 2], device=dpcpp_device)
        x_dpcpp.index_fill_(0, index, -2)

        print("x_dpcpp = ", x_dpcpp.to("cpu"))
        self.assertEqual(x, x_dpcpp.to(cpu_device))
