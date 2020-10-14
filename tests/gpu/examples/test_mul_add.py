import torch
import torch.nn as nn

import torch_ipex
from torch.testing._internal.common_utils import TestCase


dpcpp_device = torch.device("dpcpp")
cpu_device = torch.device("cpu")


class  TestTorchMethod(TestCase):
    def test_padded(self, dtype=torch.float):

        C = 2
        x = torch.randn(1, C, 3, 3)
        y = torch.randn(1, C, 3, 3)
        z = torch.randn(1, C, 3, 3)

        conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)

        a = x
        b = conv(y)
        c = conv(z)
        y_cpu = a.mul(b).add(c)
        print("cpu mul + add result", y_cpu)

        conv_dpcpp = conv.to("dpcpp")
        a_d = x.to("dpcpp")
        b_d = conv_dpcpp(y.to("dpcpp"))
        c_d = conv_dpcpp(z.to("dpcpp"))

        y_dpcpp = a_d.mul(b_d).add(c_d)
        print("dpcpp mul + add result", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_dpcpp_2 = torch_ipex.mul_add(a_d, b_d, c_d)
        print("dpcpp mul_add_ result", y_dpcpp_2.cpu())
        self.assertEqual(y_cpu, y_dpcpp_2.cpu())

    def test_expand(self, dtype=torch.float):

        C = 16
        x = torch.randn(1, C, 3, 3)
        y = torch.randn(1, C, 1, 1)
        z = torch.randn(1, C, 3, 3)

        conv1 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)

        conv1.to("dpcpp")
        conv2.to("dpcpp")

        x = conv1(x.to("dpcpp"))
        y = y.to("dpcpp")
        z = conv2(z.to("dpcpp"))
        for i in range(16):
            y[0][i][0][0] = i

        real = torch_ipex.mul_add(x, y, z).cpu()
        ref = x.cpu() * y.cpu() + z.cpu()
        print(real, ref)
        self.assertEqual(real, ref)

