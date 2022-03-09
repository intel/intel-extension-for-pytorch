import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import matplotlib.pyplot as plt
import pytest

cpu_device = torch.device('cpu')
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_cholesky_inverse(self, dtype=torch.float):
        a = torch.randn(3, 3).to(cpu_device)

        a = torch.mm(a, a.t()) + 1e-05 * torch.eye(3)  # make symmetric positive definite
        u = torch.cholesky(a)
        print("a", a)

        t = torch.cholesky_inverse(u)
        print("cpu", t)
        t_dpcpp = torch.cholesky_inverse(u.to(dpcpp_device))
        print("xpu", t_dpcpp.to(cpu_device))
        self.assertEqual(t, t_dpcpp.to(cpu_device))

        t1 = torch.cholesky_inverse(u, upper=True)
        print("cpu", t1)
        t1_dpcpp = torch.cholesky_inverse(u.to(dpcpp_device), upper=True)
        print("xpu", t1_dpcpp.to(cpu_device))
        self.assertEqual(t1, t1_dpcpp.to(cpu_device))

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_geqrf(self, dtype=torch.float):
        A = torch.tensor([[6.80, -2.11, 5.66, 5.97, 8.23],
                          [-6.05, -3.30, 5.36, -4.44, 1.08],
                          [-0.45, 2.58, -2.70, 0.27, 9.04],
                          [8.32, 2.71, 4.35, -7.17, 2.14],
                          [-9.67, -5.14, -7.26, 6.08, -6.87]]).t().to(cpu_device)

        print("A ", A.to(cpu_device))

        a, tau = torch.geqrf(A)
        print("a ", a.to(cpu_device))
        print("tau", tau.to(cpu_device))

        A = torch.tensor([[6.80, -2.11, 5.66, 5.97, 8.23],
                          [-6.05, -3.30, 5.36, -4.44, 1.08],
                          [-0.45, 2.58, -2.70, 0.27, 9.04],
                          [8.32, 2.71, 4.35, -7.17, 2.14],
                          [-9.67, -5.14, -7.26, 6.08, -6.87]]).t().to(dpcpp_device)

        print("A DPCPP", A.to(cpu_device))

        a_dpcpp, tau_dpcpp = torch.geqrf(A)
        print("a DPCPP", a_dpcpp.to(cpu_device))
        print("tau DPCPP", tau_dpcpp.to(cpu_device))

        self.assertEqual(a, a_dpcpp.to(cpu_device))
        self.assertEqual(tau, tau_dpcpp.to(cpu_device))

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_ger(self, dtype=torch.float):
        v1 = torch.arange(1., 5., device=cpu_device)
        v2 = torch.arange(1., 4., device=cpu_device)

        A12 = torch.ger(v1, v2)
        print("cpu v1 ", v1.to(cpu_device))
        print("cpu v2 ", v2.to(cpu_device))
        print("cpu A12 ", A12.to(cpu_device))

        A21 = torch.ger(v2, v1)
        print("cpu v1 ", v2.to(cpu_device))
        print("cpu v2 ", v1.to(cpu_device))
        print("cpu A21 ", A21.to(cpu_device))

        v1 = torch.arange(1., 5., device=dpcpp_device)
        v2 = torch.arange(1., 4., device=dpcpp_device)

        A12_dpcpp = torch.ger(v1, v2)
        print("dpcpp v1 ", v1.to(cpu_device))
        print("dpcpp v2 ", v2.to(cpu_device))
        print("dpcpp A12_dpcpp ", A12_dpcpp.to(cpu_device))

        A21_dpcpp = torch.ger(v2, v1)
        print("dpcpp v1 ", v2.to(cpu_device))
        print("dpcpp v2 ", v1.to(cpu_device))
        print("dpcpp A21_dpcpp ", A21_dpcpp.to(cpu_device))

        self.assertEqual(A12, A12_dpcpp.to(cpu_device))
        self.assertEqual(A21, A21_dpcpp.to(cpu_device))

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_addr(self, dtype=torch.float):
        x1_cpu = torch.randn(3, dtype=torch.float)
        x2_cpu = torch.randn(2, dtype=torch.float)
        M_cpu = torch.randn(3, 2, dtype=torch.float)
        y_cpu = torch.addr(M_cpu, x1_cpu, x2_cpu)

        x1_xpu = x1_cpu.to(dpcpp_device)
        x2_xpu = x2_cpu.to(dpcpp_device)
        M_xpu = M_cpu.to(dpcpp_device)
        y_xpu = torch.addr(M_xpu, x1_xpu, x2_xpu)

        self.assertEqual(y_cpu, y_xpu.cpu())

        y_cpu = M_cpu.addr(x1_cpu, x2_cpu)
        y_xpu = M_xpu.addr(x1_xpu, x2_xpu)
        self.assertEqual(y_cpu, y_xpu.cpu())

        M_cpu.addr_(x1_cpu, x2_cpu)
        M_xpu.addr_(x1_xpu, x2_xpu)
        self.assertEqual(M_cpu, M_xpu.cpu())
