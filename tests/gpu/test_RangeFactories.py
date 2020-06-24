import numpy
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase
import torch_ipex


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("dpcpp")


class TestTorchMethod(TestCase):
    def test_RangeFactories(self, dtype=torch.float):

        #x=torch.tensor([1,1,1,1,1], device=cpu_device)
        x = torch.logspace(start=-10, end=10, steps=5, device=cpu_device)
        y = torch.linspace(start=-10, end=10, steps=5, device=cpu_device)
        z = torch.arange(1, 2.5, 0.5, device=cpu_device)
        n = torch.range(1, 2.5, 0.5, device=cpu_device)

        # x_dpcpp=x.to("dpcpp")
        x_out = torch.logspace(start=-10, end=10, steps=5, device=dpcpp_device)
        y_out = torch.linspace(start=-10, end=10, steps=5, device=dpcpp_device)
        z_out = torch.arange(1, 2.5, 0.5, device=dpcpp_device)
        n_out = torch.range(1, 2.5, 0.5, device=dpcpp_device)

        print("cpu: ")
        print(x)
        print(y)
        print(z)
        print(n)

        print("dpcpp: ")
        print(x_out.to("cpu"))
        print(y_out.to("cpu"))
        print(z_out.to("cpu"))
        print(n_out.to("cpu"))
        self.assertEqual(x, x_out.cpu(), 1e-2)
        self.assertEqual(y, y_out.cpu())
        self.assertEqual(z, z_out.cpu())
        self.assertEqual(n, n_out.cpu())
