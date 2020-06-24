import numpy
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestNNMethod(TestCase):
    def test_random_norm(self, dtype=torch.float):

        # note dpcpp backend only support float or double
        x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=cpu_device,
                         dtype=torch.float).view(4, 2)

        y = x.to("dpcpp")
        x_out = torch.roll(x, shifts=(2, 1), dims=(0, 1))
        y_out = torch.roll(y, shifts=(2, 1), dims=(0, 1))
        print("cpu:")
        print(x_out)
        print("dpcpp:")
        print(y_out.to("cpu"))
        self.assertEqual(x_out, y_out.cpu())

        x_out = torch.roll(x, 1, 0)
        y_out = torch.roll(y, 1, 0)
        print("cpu:")
        print(x_out)
        print("dpcpp:")
        print(y_out.to("cpu"))
        self.assertEqual(x_out, y_out.cpu())
