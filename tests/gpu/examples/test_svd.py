import time

import torch
from torch.testing._internal.common_utils import TestCase

import ipex

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_svd(self, dtype=torch.float):
        # Since U and V of an SVD is not unique, each vector can be multiplied by an arbitrary phase factor e^iϕ
        # while the SVD result is still correct. Different platforms, like Numpy, or inputs on different device types,
        # may produce different U and V tensors.

        a = torch.randn(5, 5)
        a_xpu = a.to('xpu')

        u, s, v = torch.svd(a)
        # print(u)
        # print(s)
        # print(v)
        r_cpu = u * s * v
        # print(r_cpu)

        u_xpu, s_xpu, v_xpu = torch.svd(a_xpu)
        # print(u_xpu.cpu())
        # print(s_xpu.cpu())
        # print(v_xpu.cpu())
        r_xpu = u_xpu * s_xpu * v_xpu
        # print(r_xpu.cpu())

        self.assertEqual(r_cpu, r_xpu.cpu())

    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_batch_svd(self, dtype=torch.float):
        # Since U and V of an SVD is not unique, each vector can be multiplied by an arbitrary phase factor e^iϕ
        # while the SVD result is still correct. Different platforms, like Numpy, or inputs on different device types,
        # may produce different U and V tensors.
        a = torch.randn(5, 5, 5)
        a_xpu = a.to('xpu')

        u, s, v = torch.svd(a)
        # print(u)
        # print(s)
        # print(v)
        r_cpu = u * s * v

        u_xpu, s_xpu, v_xpu = torch.svd(a_xpu)
        # print(u_xpu.cpu())
        # print(s_xpu.cpu())
        # print(v_xpu.cpu())
        r_xpu = u_xpu * s_xpu * v_xpu

        self.assertEqual(r_cpu, r_xpu.cpu())
