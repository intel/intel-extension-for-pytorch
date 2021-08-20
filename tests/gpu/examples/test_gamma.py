import torch
from torch.testing._internal.common_utils import TestCase
import ipex
import numpy as np
import pytest


cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not ipex._onemkl_is_enabled()")
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

    @pytest.mark.skipif("not ipex._onemkl_is_enabled()")
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


    def test_polygamma(self, dtype=torch.float):
        x_cpu = torch.tensor([1, 0.5])
        x_xpu = x_cpu.to('xpu')

        for n in range(5):
            print("n = ", n)
            y_cpu = torch.polygamma(n, x_cpu)
            y_xpu = torch.polygamma(n, x_xpu)
            print("y_cpu = ", y_cpu)
            print("y_xpu = ", y_xpu.cpu())
            self.assertEqual(y_cpu, y_xpu)

            y_cpu = x_cpu.polygamma(n)
            y_xpu = x_xpu.polygamma(n)
            print("y_cpu = ", y_cpu)
            print("y_xpu = ", y_xpu.cpu())
            self.assertEqual(y_cpu, y_xpu)

            x_cpu_clone = x_cpu.clone()
            x_xpu_clone = x_xpu.clone()
            x_cpu_clone.polygamma_(n)
            x_xpu_clone.polygamma_(n)
            print("x_cpu_clone = ", x_cpu_clone)
            print("x_xpu_clone = ", x_xpu_clone.cpu())
            self.assertEqual(x_cpu_clone, x_xpu_clone)

