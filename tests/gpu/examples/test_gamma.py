import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import numpy as np
import pytest


class TestTorchMethod(TestCase):
    def test_lgamma(self, dtype=torch.float):
        a = np.array([[0.5, 1, 1.5],
                      [2.5, 3, 3.5]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        x_dpcpp = x.to("xpu")

        y = torch.lgamma(x)
        y_dpcpp = torch.lgamma(x_dpcpp)
        y_dpcpp_lgamma = x_dpcpp.lgamma()

        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp_lgamma.cpu())

        x.lgamma_()
        x_dpcpp.lgamma_()

        self.assertEqual(x, x_dpcpp.cpu())

    def test_lgamma_bf16(self, dtype=torch.bfloat16):
        a = np.array([[0.5, 1, 1.5],
                      [2.5, 3, 3.5]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        x_dpcpp = x.to("xpu")

        y = torch.lgamma(x)
        y_dpcpp = torch.lgamma(x_dpcpp)
        y_dpcpp_lgamma = x_dpcpp.lgamma()

        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp_lgamma.cpu())

        x.lgamma_()
        x_dpcpp.lgamma_()

        self.assertEqual(x, x_dpcpp.cpu())

    def test_lgamma_out(self, dtype=torch.float):
        a = np.array([[0.5, 1, 1.5],
                      [2.5, 3, 3.5]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        c_result = torch.zeros_like(x)
        x_dpcpp = x.to("xpu")
        x_result = torch.zeros_like(x_dpcpp)

        y = torch.lgamma(x, out=c_result)
        y_dpcpp = torch.lgamma(x_dpcpp, out=x_result)

        self.assertEqual(y, y_dpcpp.cpu())

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_mvlgamma(self, dtype=torch.float):
        a = np.array([[1.6835, 1.8474, 1.1929],
                      [1.0475, 1.7162, 1.4180]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        x_dpcpp = x.to("xpu")

        y = torch.mvlgamma(x, 2)
        y_dpcpp = torch.mvlgamma(x_dpcpp, 2)
        y_dpcpp_mvlgamma = x_dpcpp.mvlgamma(2)
        self.assertEqual(y, y_dpcpp.cpu())
        self.assertEqual(y, y_dpcpp_mvlgamma.cpu())

        x.mvlgamma_(2)
        x_dpcpp.mvlgamma_(2)
        self.assertEqual(x, x_dpcpp.cpu())

    def test_mvlgamma_out(self, dtype=torch.float):
        a = np.array([[4.5, 2, 1.5],
                      [2.5, 3, 3.5]])
        data = torch.from_numpy(a)
        x = data.clone().detach()
        c_result = torch.zeros_like(x)
        x_dpcpp = x.to("xpu")
        x_result = torch.zeros_like(x_dpcpp)

        y = torch.mvlgamma(x, 2, out=c_result)
        y_dpcpp = torch.mvlgamma(x_dpcpp, 2, out=x_result)

        self.assertEqual(y, y_dpcpp.cpu())


    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_polygamma(self, dtype=torch.float):
        x_cpu = torch.tensor([1, 0.5])
        x_xpu = x_cpu.to('xpu')

        for n in range(5):
            y_cpu = torch.polygamma(n, x_cpu)
            y_xpu = torch.polygamma(n, x_xpu)
            self.assertEqual(y_cpu, y_xpu)

            y_cpu = x_cpu.polygamma(n)
            y_xpu = x_xpu.polygamma(n)
            self.assertEqual(y_cpu, y_xpu)

            x_cpu_clone = x_cpu.clone()
            x_xpu_clone = x_xpu.clone()
            x_cpu_clone.polygamma_(n)
            x_xpu_clone.polygamma_(n)
            self.assertEqual(x_cpu_clone, x_xpu_clone)
