import torch
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_index_copy(self, dtype=torch.float):

        x = torch.ones([5, 3], device=cpu_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_copy_(0, index, t)

        x_xpu = torch.ones([5, 3], device=xpu_device)
        index_xpu = torch.tensor([0, 4, 2], device=xpu_device)
        t_d = t.to("xpu")
        x_xpu.index_copy_(0, index_xpu, t_d)

        self.assertEqual(x, x_xpu.to(cpu_device))
