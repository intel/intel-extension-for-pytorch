import torch
import torch.nn as nn
import ipex
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestNNMethod(TestCase):
    def test_linear_relu(self, dtype=torch.float):
        linear = nn.Linear(4, 2).to("xpu")
        linear_relu = ipex.xpu.LinearReLU(4, 2).to("xpu")
        linear_relu.weight = linear.weight
        linear_relu.bias = linear.bias
        x = torch.tensor([[1.23, 2.34, 6.45, 2.22], [0.23, 1.34, 7.45, 1.22]], requires_grad=True, device=dpcpp_device, dtype=dtype)
        y = nn.ReLU()(linear(x).cpu())
        y_dpcpp = linear_relu(x)

        # no fuse
        print("input", x.to("cpu"))
        print("weight", linear.weight.to("cpu"))
        print("bias", linear.bias.to("cpu"))
        print("no fuse", nn.ReLU()(linear(x)).to("cpu"))

        print()
        print("input", x.to("cpu"))
        print("weight", linear_relu.weight.to("cpu"))
        print("bias", linear_relu.bias.to("cpu"))
        print("fuse", linear_relu(x).to("cpu"))
        self.assertEqual(y, y_dpcpp.cpu())
