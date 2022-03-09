import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

dpcpp_device = torch.device("xpu")
cpu_device = torch.device("cpu")


class TestTorchMethod(TestCase):
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

        conv_dpcpp = conv.to("xpu")
        a_d = x.to("xpu")
        b_d = conv_dpcpp(y.to("xpu"))
        c_d = conv_dpcpp(z.to("xpu"))

        y_dpcpp = a_d.mul(b_d).add(c_d)
        print("dpcpp mul + add result", y_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())

        y_dpcpp_2 = torch.xpu.intrinsic.MulAdd(a_d, b_d, c_d)
        print("dpcpp MulAdd result", y_dpcpp_2.cpu())
        self.assertEqual(y_cpu, y_dpcpp_2.cpu())

    def test_expand(self, dtype=torch.float):

        C = 16
        x = torch.randn(1, C, 3, 3)
        y = torch.randn(1, C, 1, 1)
        z = torch.randn(1, C, 3, 3)

        conv1 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)
        conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False)

        conv1.to("xpu")
        conv2.to("xpu")

        x = conv1(x.to("xpu"))
        y = y.to("xpu")
        z = conv2(z.to("xpu"))
        for i in range(16):
            y[0][i][0][0] = i

        real = torch.xpu.intrinsic.MulAdd(x, y, z).cpu()
        ref = x.cpu() * y.cpu() + z.cpu()
        print(real, ref)
        self.assertEqual(real, ref)
