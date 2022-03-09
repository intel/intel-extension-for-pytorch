import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import numpy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_threshold(self, dtype=torch.float):

        # functionality
        x_ref = torch.ones([2, 2], device=cpu_device)
        x_ref[0][0] = 1
        x_ref[0][1] = 3
        x_ref[1][0] = 2
        x_ref[1][1] = 1

        y_ref = nn.Threshold(2, 0)(x_ref)
        print(y_ref)
        x = x_ref.to("xpu")
        y = nn.Threshold(2, 0)(x)
        print(y.to("cpu"))
        self.assertEqual(y_ref, y.cpu())
