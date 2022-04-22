import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_index_copy_dim_0(self, dtype=torch.float):
        x = torch.ones([5, 3], device=cpu_device)
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_copy_(0, index, t)

        x_xpu = torch.ones([5, 3], device=xpu_device)
        index_xpu = torch.tensor([0, 4, 2], device=xpu_device)
        t_d = t.to("xpu")
        x_xpu.index_copy_(0, index_xpu, t_d)

        self.assertEqual(x, x_xpu.to(cpu_device))

    def test_index_copy_dim_1(self, dtype=torch.float):
        x = torch.zeros([3, 5], device=cpu_device)
        t = torch.tensor([[1, 2, 3], [6, 7, 8], [11, 12, 13]], dtype=torch.float)
        index = torch.tensor([0, 4, 2])
        x.index_copy_(1, index, t)

        x_xpu = torch.zeros([3, 5], device=xpu_device)
        index_xpu = torch.tensor([0, 4, 2], device=xpu_device)
        t_d = t.to("xpu")
        x_xpu.index_copy_(1, index_xpu, t_d)

        self.assertEqual(x, x_xpu.to(cpu_device))
