import numpy
import torch
import torch.nn as nn
import torch_ipex
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
dpcpp_device = torch.device('dpcpp')


class TestTorchMethod(TestCase):
    def test_math_compare(self, dtype=torch.float):

        x = torch.tensor([[True, True], [True, True]], device=cpu_device)
        y = torch.tensor([[1, 2], [3, 4]], device=cpu_device)
        z = torch.tensor([[1, 1], [4, 4]], device=cpu_device)
        x_dpcpp = x.to("xpu")
        y_dpcpp = y.to("xpu")
        z_dpcpp = z.to("xpu")
        x = torch.lt(y, z)
        x2 = torch.eq(y, z)
        x3 = torch.gt(y, z)
        x4 = torch.le(y, z)
        x5 = torch.ne(y, z)
        x6 = torch.ge(y, z)

        print("cpu: ")
        print(x)
        print(x2)
        print(x3)
        print(x4)
        print(x5)
        print(x6)
        print(y.lt_(z))
        print(y.eq_(z))
        print(y.gt_(z))
        print(y.le_(z))
        print(y.ne_(z))
        print(y.ge_(z))

        x_out = torch.lt(y_dpcpp, z_dpcpp)
        x2_out = torch.eq(y_dpcpp, z_dpcpp)
        x3_out = torch.gt(y_dpcpp, z_dpcpp)
        x4_out = torch.le(y_dpcpp, z_dpcpp)
        x5_out = torch.ne(y_dpcpp, z_dpcpp)
        x6_out = torch.ge(y_dpcpp, z_dpcpp)

        print("dpcpp: ")
        print(x_out.to("cpu"))
        print(x2_out.to("cpu"))
        print(x3_out.to("cpu"))
        print(x4_out.to("cpu"))
        print(x5_out.to("cpu"))
        print(x6_out.to("cpu"))
        print(y_dpcpp.lt_(z_dpcpp).to("cpu"))
        print(y_dpcpp.eq_(z_dpcpp).to("cpu"))
        print(y_dpcpp.gt_(z_dpcpp).to("cpu"))
        print(y_dpcpp.le_(z_dpcpp).to("cpu"))
        print(y_dpcpp.ne_(z_dpcpp).to("cpu"))
        print(y_dpcpp.ge_(z_dpcpp).to("cpu"))
        self.assertEqual(x, x_out.cpu())
        self.assertEqual(x2, x2_out.cpu())
        self.assertEqual(x3, x3_out.cpu())
        self.assertEqual(x4, x4_out.cpu())
        self.assertEqual(x5, x5_out.cpu())
        self.assertEqual(x6, x6_out.cpu())
        self.assertEqual(y.lt_(z), y_dpcpp.lt_(z_dpcpp).to("cpu"))
        self.assertEqual(y.eq_(z), y_dpcpp.eq_(z_dpcpp).to("cpu"))
        self.assertEqual(y.gt_(z), y_dpcpp.gt_(z_dpcpp).to("cpu"))
        self.assertEqual(y.le_(z), y_dpcpp.le_(z_dpcpp).to("cpu"))
        self.assertEqual(y.ne_(z), y_dpcpp.ne_(z_dpcpp).to("cpu"))
        self.assertEqual(y.ge_(z), y_dpcpp.ge_(z_dpcpp).to("cpu"))
