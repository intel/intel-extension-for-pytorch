import torch
from torch.testing._internal.common_utils import TestCase
import torch_ipex
import numpy as np
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch_ipex._onemkl_is_enabled()")
    def test_lgamma(self, dtype=torch.float):
        a = np.array([[0.5, 1, 1.5],
                      [2.5, 3, 3.5]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        x_dpcpp = x.to(dpcpp_device)

        y = torch.lgamma(x)
        y_dpcpp = torch.lgamma(x_dpcpp)
        y_dpcpp_lgamma = x_dpcpp.lgamma()

        print("x: ")
        print(x)
        print("y: ")
        print(y)
        print("y_dpcpp: ")
        print(y_dpcpp.to(cpu_device))
        print("y_dpcpp_lgamma: ")
        print(y_dpcpp_lgamma.to(cpu_device))
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        self.assertEqual(y, y_dpcpp_lgamma.to(cpu_device))


        print("---")
        x.lgamma_()
        x_dpcpp.lgamma_()
        print("x: ")
        print(x)
        print("x_dpcpp: ")
        print(x_dpcpp.to(cpu_device))
        self.assertEqual(x, x_dpcpp.to(cpu_device))

    @pytest.mark.skip()
    def test_mvlgamma(self, dtype=torch.float):
        a = np.array([[1.6835, 1.8474, 1.1929],
                      [1.0475, 1.7162, 1.4180]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        x_dpcpp = x.to(dpcpp_device)

        y = torch.mvlgamma(x, 2)
        y_dpcpp = torch.mvlgamma(x_dpcpp, 2)
        y_dpcpp_mvlgamma = x_dpcpp.mvlgamma(2)

        print("x: ")
        print(x)
        print("y: ")
        print(y)
        print("y_dpcpp: ")
        print(y_dpcpp.to(cpu_device))
        print("y_dpcpp_mvlgamma: ")
        print(y_dpcpp_mvlgamma.to(cpu_device))
        self.assertEqual(y, y_dpcpp.to(cpu_device))
        self.assertEqual(y, y_dpcpp_mvlgamma.to(cpu_device))

        print("---")
        x.mvlgamma_(2)
        x_dpcpp.mvlgamma_(2)
        print("x: ")
        print(x)
        print("x_dpcpp: ")
        print(x_dpcpp.to(cpu_device))
        self.assertEqual(x, x_dpcpp.to(cpu_device))


