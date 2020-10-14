import torch
import torch_ipex
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device('cpu')
dpcpp_device = torch.device('dpcpp')


class TestTorchMethod(TestCase):
    def test_masked_select(self, dtype=torch.float):
        x = torch.randn(3, 4, dtype=torch.float, device=torch.device("cpu"))
        x_mask = x.ge(0.5)

        print("x", x)
        print("mask", x_mask)
        print("cpu masked_select", torch.masked_select(x, x_mask))

        y = x.to("dpcpp")
        y_mask = x_mask.to("dpcpp")
        print("dpcpp masked_select", torch.masked_select(y, y_mask).cpu())
