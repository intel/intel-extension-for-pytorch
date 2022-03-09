import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import numpy

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_resize(self, dtype=torch.float):

        x = torch.ones([2, 2, 4, 3], device=dpcpp_device, dtype=dtype)
        x.resize_(1, 2, 3, 4)

        y = torch.ones([2, 2, 4, 3], device=cpu_device, dtype=dtype)
        y.resize_(1, 2, 3, 4)

        print("dpcpp: ")
        print(x.to("cpu"))
        print("cpu: ")
        print(y)
        self.assertEqual(y, x.cpu())
