import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_add(self, dtype=torch.float):
        test = torch.tensor(3, device=dpcpp_device)
        test.item()

        x = torch.ones([5, 3], device=cpu_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_add_(0, index, t)
        print("x = ", x)

        x_dpcpp = torch.ones([5, 3], device=dpcpp_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         dtype=torch.float, device=dpcpp_device)
        index = torch.tensor([0, 4, 2], device=dpcpp_device)
        x_dpcpp.index_add_(0, index, t)
        print("x_dpcpp = ", x_dpcpp.to("cpu"))
        self.assertEqual(x, x_dpcpp.to(cpu_device))

# x_dpcpp = torch.ones([3,5], device = dpcpp_device)
# t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device = dpcpp_device)
# index = torch.tensor([0, 4, 2], device = dpcpp_device)
# x_dpcpp.index_add_(1, index, t)
# print("x_dpcpp = ", x_dpcpp.to("cpu"))

# x_dpcpp = torch.ones([5,1], device = dpcpp_device)
# t = torch.tensor([[100], [100], [100], [100], [100]], dtype=torch.float, device = dpcpp_device)
# index = torch.tensor([0], device = dpcpp_device)
# x_dpcpp.index_add_(1, index, t)
# print("x_dpcpp = ", x_dpcpp.to("cpu"))
