import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


@pytest.mark.skip(reason="Skip due to failing in oneDNN acceptance test only")
class TestTorchMethod(TestCase):
    def test_einsum(self, dtype=torch.float):
        x_cpu1 = torch.randn(5, dtype=dtype, device=cpu_device)
        x_cpu2 = torch.randn(4, dtype=dtype, device=cpu_device)
        y_cpu = torch.einsum('i,j->ij', x_cpu1, x_cpu2)
        x_dpcpp1 = x_cpu1.to(dpcpp_device)
        x_dpcpp2 = x_cpu2.to(dpcpp_device)
        y_dpcpp = torch.einsum('i,j->ij', x_dpcpp1, x_dpcpp2)
        print('y_cpu = ', y_cpu)
        print('y_dpcpp = ', y_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

        A_cpu = torch.randn(3, 5, 4, dtype=dtype, device=cpu_device)
        l_cpu = torch.randn(2, 5, dtype=dtype, device=cpu_device)
        r_cpu = torch.randn(2, 4, dtype=dtype, device=cpu_device)
        y_cpu = torch.einsum('bn,anm,bm->ba', l_cpu, A_cpu, r_cpu)
        A_dpcpp = A_cpu.to(dpcpp_device)
        l_dpcpp = l_cpu.to(dpcpp_device)
        r_dpcpp = r_cpu.to(dpcpp_device)
        y_dpcpp = torch.einsum('bn,anm,bm->ba', l_dpcpp, A_dpcpp, r_dpcpp)
        print('y_cpu = ', y_cpu)
        print('y_dpcpp = ', y_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

        As_cpu = torch.randn(3, 2, 5, dtype=dtype, device=cpu_device)
        Bs_cpu = torch.randn(3, 5, 4, dtype=dtype, device=cpu_device)
        y_cpu = torch.einsum('bij,bjk->bik', As_cpu, Bs_cpu)
        As_dpcpp = As_cpu.to(dpcpp_device)
        Bs_dpcpp = Bs_cpu.to(dpcpp_device)
        y_dpcpp = torch.einsum('bij,bjk->bik', As_dpcpp, Bs_dpcpp)
        print('y_cpu = ', y_cpu)
        print('y_dpcpp = ', y_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

        A_cpu = torch.randn(3, 3, dtype=dtype, device=cpu_device)
        y_cpu = torch.einsum('ii->i', A_cpu)
        A_dpcpp = A_cpu.to(dpcpp_device)
        y_dpcpp = torch.einsum('ii->i', A_dpcpp)
        print('y_cpu = ', y_cpu)
        print('y_dpcpp = ', y_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))  # pr

        A_cpu = torch.randn(4, 3, 3, dtype=dtype, device=cpu_device)
        y_cpu = torch.einsum('...ii->...i', A_cpu)
        A_dpcpp = A_cpu.to(dpcpp_device)
        y_dpcpp = torch.einsum('...ii->...i', A_dpcpp)
        print('y_cpu = ', y_cpu)
        print('y_dpcpp = ', y_dpcpp.to(cpu_device))
        self.assertEqual(y_cpu, y_dpcpp.to(cpu_device))

        A_dpcpp = torch.randn(2, 3, 4, 5, dtype=dtype, device=dpcpp_device)
        self.assertEqual(torch.einsum('...ij->...ji', A_dpcpp).shape, torch.Size([2, 3, 5, 4]))
