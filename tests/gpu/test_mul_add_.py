import torch
import torch.nn as nn

import torch_ipex
from torch.testing._internal.common_utils import TestCase


dpcpp_device = torch.device("dpcpp")
cpu_device = torch.device("cpu")


class  TestTorchMethod(TestCase):
    def test_admm(self, dtype=torch.float):

        a = torch.randn([3, 4], dtype=dtype)
        b = torch.randn([3, 4], dtype=dtype)
        c = torch.ones([3, 4], dtype=dtype)

        a_d = a.to(dpcpp_device)
        b_d = b.to(dpcpp_device)
        c_d = c.to(dpcpp_device)

        y = a.mul(b).add(c)
        print("cpu mul + add result", y)

        y_d = a_d.mul(b_d).add(c_d)
        print("dpcpp mul + add result", y_d.cpu())
        self.assertEqual(y, y_d.cpu())

        z = torch_ipex.mul_add(a_d, b_d, c_d)
        print("dpcpp mul_add_ result", c_d.cpu())
        self.assertEqual(y, z.cpu())
