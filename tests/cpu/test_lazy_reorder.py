"""Tests for lazy reorder."""
from __future__ import division
from __future__ import print_function

import os
import math
import time
import random
import unittest
from functools import reduce

import torch
import _torch_ipex as ipex
ipex._initialize_aten_bindings()

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch._six import inf, nan

from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, brute_pdist, brute_cdist, slowTest, \
    skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf

def get_rand_seed():
    return int(time.time() * 1000000000)

device = torch.device("dpcpp:0")
class TestConv(TestCase):
    def test_Conv2d_with_cpu(self):
        rand_seed = int(get_rand_seed())
        print("test_Conv2d_with_cpu rand sed: {}".format(rand_seed))
        torch.manual_seed(rand_seed)
        conv_cpu = torch.nn.Conv2d(1, 1, (3, 3))
        conv_dpcpp = torch.nn.Conv2d(1, 1, (3, 3)).to(device=device)

        conv_dpcpp.weight.data = conv_cpu.weight.data.to(device=device)
        conv_dpcpp.bias.data = conv_cpu.bias.data.to(device=device)

        input_cpu = torch.rand((1, 1, 7, 7))
        input_dpcpp = input_cpu.to(device=device)

        ipex.enable_auto_dnnl()
        out_dpcpp = conv_dpcpp(input_dpcpp)

        ipex.disable_auto_dnnl()
        out_dpcpp_cpu = out_dpcpp.to('cpu')
        out_cpu = conv_cpu(input_cpu)
        self.assertEqual(out_dpcpp.size(), out_cpu.size())
        self.assertEqual(out_cpu, out_dpcpp_cpu)

    def _seq_conf(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        conv_dpcpp1 = torch.nn.Conv2d(1, 1, (7, 7)).to(device=device)
        conv_dpcpp2 = torch.nn.Conv2d(1, 1, (5, 5)).to(device=device)
        conv_dpcpp3 = torch.nn.Conv2d(1, 1, (3, 3)).to(device=device)
        input_cpu = torch.rand((1, 1, 105, 105))
        input_dpcpp = input_cpu.to(device=device)

        out_dpcpp1 = conv_dpcpp1(input_dpcpp)
        out_dpcpp2 = conv_dpcpp2(out_dpcpp1)
        out_dpcpp3 = conv_dpcpp3(out_dpcpp2)
        return out_dpcpp3


    def test_seq_conv(self):
        ipex.disable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("test_seq_conv rand sed: {}".format(rand_seed))
        res_cpu = self._seq_conf('cpu', rand_seed)

        ipex.enable_auto_dnnl()
        res_dpcpp = self._seq_conf(device, rand_seed)
        self.assertEqual(res_cpu, res_dpcpp.to('cpu'))

class TestBinaryOp(TestCase):
    def test_dil_add(self):
        ipex.enable_auto_dnnl()
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        alpha = torch.randn(1, dtype=torch.float32).item()

        x_cpu = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y_cpu = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        y_dpcpp = y_cpu.to(device=device)

        # add
        self.assertEqual(
            x_cpu + y_cpu,
            x_dpcpp + y_dpcpp)

        self.assertEqual(
            torch.add(x_cpu, y_cpu, alpha=alpha),
            torch.add(x_dpcpp, y_dpcpp, alpha=alpha))

        # add_out
        out_cpu = x_cpu.clone()
        out_dpcpp = out_cpu.to(device=device)
        torch.add(x_cpu, y_cpu, alpha=alpha, out=out_cpu)
        torch.add(x_dpcpp, y_dpcpp, alpha=alpha, out=out_dpcpp)
        self.assertEqual(out_cpu, out_dpcpp)

    def _test_dil_add(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((8, 8)).to(device=device)
        a1 = a[0:2, :]
        a2 = a[4:6, :]
        self.assertEqual(a1.is_contiguous(), True)
        self.assertEqual(a2.is_contiguous(), True)
        a1 += a2
        return a1

    def test_dil_add_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("test_dil_add_ rand sed: {}".format(rand_seed))
        res_dcpp_dnnl = self._test_dil_add("dpcpp:0", rand_seed)

        ipex.disable_auto_dnnl()
        res_dcpp_cpu = self._test_dil_add("dpcpp:0", rand_seed)

        res_cpu = self._test_dil_add("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

    def test_dil_add_scalar(self):
        ipex.enable_auto_dnnl()
        a = torch.rand((8, 8)).to(device=device)
        a += 2

    def test_add_out(self):
        ipex.enable_auto_dnnl()
        a = torch.rand((8, 8)).to(device=device)
        b = torch.rand((8, 8)).to(device=device)
        c = torch.rand((8, 8)).to(device=device)
        c = a + b

    def _test_mul_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((20, 20)).to(device=device)
        b = torch.rand((20, 20)).to(device=device)
        a.mul_(b)
        return a

    def test_mul_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("test_mul_ rand sed: {}".format(rand_seed))
        a1 = self._test_mul_(device, rand_seed)
        a2 = self._test_mul_('cpu', rand_seed)
        self.assertEqual(a2, a1.to('cpu'))

    def _test_relu_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((30, 30)).to(device=device)
        a.relu_()
        return a

    def test_relu_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("test_relu_ rand sed: {}".format(rand_seed))
        a1 = self._test_relu_(device, rand_seed)
        a2 = self._test_relu_('cpu', rand_seed)
        self.assertEqual(a2, a1.to('cpu'))


class TestMixOp(TestCase):
    def _test_conv_add_relu_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=device)
        conv_op_input = torch.rand((1, 1, 10, 10)).to(device=device).requires_grad_(True)
        conv_op_output = conv_op(conv_op_input)
        add_src = torch.rand((1, 1, 4, 4)).to(device=device).requires_grad_(True)
        conv_op_output += add_src
        conv_op_output.relu_()

        return conv_op_output, conv_op_input, add_src

    def test_conv_add_relu_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("test_dil_add_ rand sed: {}".format(rand_seed))
        res_dcpp_dnnl, input_dpcpp_dnnl, _ = self._test_conv_add_relu_("dpcpp:0", rand_seed)

        ipex.disable_auto_dnnl()
        res_dcpp_cpu, input_dpcpp_cpu, _ = self._test_conv_add_relu_("dpcpp:0", rand_seed)

        res_cpu, input_cpu, _ = self._test_conv_add_relu_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

        ipex.enable_auto_dnnl()
        res_dcpp_dnnl.sum().backward()
        res_dcpp_cpu.sum().backward()
        res_cpu.sum().backward()

        self.assertEqual(input_dpcpp_dnnl.grad.to('cpu'), input_cpu.grad, prec=0.0)
        self.assertEqual(input_dpcpp_cpu.grad.to('cpu'), input_cpu.grad, prec=0.0)

class TestLinearAlgebraOps(TestCase):
    def test_mm(self):
        M, N, O = 23, 8, 12
        b1_cpu = torch.randn(M, N, dtype=torch.float32)
        b2_cpu = torch.randn(N, O, dtype=torch.float32)
        b1_dpcpp = b1_cpu.to(device=device)
        b2_dpcpp = b2_cpu.to(device=device)

        # mm
        mm_cpu = torch.mm(b1_cpu, b2_cpu)
        mm_dpcpp = torch.mm(b1_dpcpp, b2_dpcpp)
        self.assertEqual(mm_cpu, mm_dpcpp)

        # mm_out
        y_cpu = torch.randn(M, O, dtype=torch.float32)
        y_dpcpp = y_cpu.to(device=device)
        torch.mm(b1_cpu, b2_cpu, out=y_cpu)
        torch.mm(b1_dpcpp, b2_dpcpp, out=y_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)

    def test_bmm(self):
        num_batches = 10
        M, N, O = 23, 8, 12
        b1_cpu = torch.randn(num_batches, M, N, dtype=torch.float32)
        b2_cpu = torch.randn(num_batches, N, O, dtype=torch.float32)
        b1_dpcpp = b1_cpu.to(device=device)
        b2_dpcpp = b2_cpu.to(device=device)

        # bmm
        bmm_cpu = torch.bmm(b1_cpu, b2_cpu)
        bmm_dpcpp = torch.bmm(b1_dpcpp, b2_dpcpp)
        self.assertEqual(bmm_cpu, bmm_dpcpp)

        # bmm_out
        y_cpu = torch.randn(num_batches, M, O, dtype=torch.float32)
        y_dpcpp = y_cpu.to(device=device)
        torch.bmm(b1_cpu, b2_cpu, out=y_cpu)
        torch.bmm(b1_dpcpp, b2_dpcpp, out=y_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)

    def test_addmm(self):
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                M, N, O = 23, 8, 12
                b1_cpu = torch.randn(M, N, dtype=torch.float32)
                b2_cpu = torch.randn(N, O, dtype=torch.float32)
                res_cpu = torch.randn(M, O, dtype=torch.float32)
                b1_dpcpp = b1_cpu.to(device=device)
                b2_dpcpp = b2_cpu.to(device=device)
                res_dpcpp = res_cpu.to(device=device)

                addmm_cpu = torch.addmm(input=res_cpu, mat1=b1_cpu, mat2=b2_cpu, alpha=alpha, beta=beta)
                addmm_dpcpp = torch.addmm(input=res_dpcpp, mat1=b1_dpcpp, mat2=b2_dpcpp, alpha=alpha, beta=beta)
                self.assertEqual(addmm_cpu, addmm_dpcpp)

                y_cpu = torch.randn(M, O, dtype=torch.float32)
                y_dpcpp = y_cpu.to(device=device)
                torch.addmm(input=res_cpu, mat1=b1_cpu, mat2=b2_cpu, alpha=alpha, beta=beta, out=y_cpu)
                torch.addmm(input=res_dpcpp, mat1=b1_dpcpp, mat2=b2_dpcpp, alpha=alpha, beta=beta, out=y_dpcpp)
                self.assertEqual(y_cpu, y_dpcpp)

    def test_addbmm(self):
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                num_batches = 10
                M, N, O = 23, 8, 12
                b1_cpu = torch.randn(num_batches, M, N, dtype=torch.float32)
                b2_cpu = torch.randn(num_batches, N, O, dtype=torch.float32)
                res_cpu = torch.randn(M, O, dtype=torch.float32)
                b1_dpcpp = b1_cpu.to(device=device)
                b2_dpcpp = b2_cpu.to(device=device)
                res_dpcpp = res_cpu.to(device=device)

                addbmm_cpu = torch.addbmm(res_cpu, b1_cpu, b2_cpu, beta=beta, alpha=alpha)
                addbmm_dpcpp = torch.addbmm(res_dpcpp, b1_dpcpp, b2_dpcpp, beta=beta, alpha=alpha)
                self.assertEqual(addbmm_cpu, addbmm_dpcpp)
                y_cpu = torch.randn(M, O, dtype=torch.float32)
                y_dpcpp = y_cpu.to(device=device)
                torch.addbmm(res_cpu, b1_cpu, b2_cpu, beta=beta, alpha=alpha, out=y_cpu)
                torch.addbmm(res_dpcpp, b1_dpcpp, b2_dpcpp, beta=beta, alpha=alpha, out=y_dpcpp)
                self.assertEqual(y_cpu, y_dpcpp)

    def test_baddbmm(self):
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                num_batches = 10
                M, N, O = 23, 8, 12
                b1_cpu = torch.randn(num_batches, M, N, dtype=torch.float32)
                b2_cpu = torch.randn(num_batches, N, O, dtype=torch.float32)
                res_cpu = torch.randn(num_batches, M, O, dtype=torch.float32)
                b1_dpcpp = b1_cpu.to(device=device)
                b2_dpcpp = b2_cpu.to(device=device)
                res_dpcpp = res_cpu.to(device=device)

                baddbmm_cpu = torch.baddbmm(res_cpu, b1_cpu, b2_cpu, alpha=alpha, beta=beta)
                baddbmm_dpcpp = torch.baddbmm(res_dpcpp, b1_dpcpp, b2_dpcpp, alpha=alpha, beta=beta)
                self.assertEqual(baddbmm_cpu, baddbmm_dpcpp)
                y_cpu = torch.randn(num_batches, M, O, dtype=torch.float32)
                y_dpcpp = y_cpu.to(device=device)
                torch.baddbmm(res_cpu, b1_cpu, b2_cpu, alpha=alpha, beta=beta, out=y_cpu),
                torch.baddbmm(res_dpcpp, b1_dpcpp, b2_dpcpp, alpha=alpha, beta=beta, out=y_dpcpp),
                self.assertEqual(y_cpu, y_dpcpp)

class TestLinear(TestCase):
    def test_linear(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x_cpu = torch.randn(3, in_features, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias).float()
            self.assertEqual(linear(x_cpu), linear(x_dpcpp))

if __name__ == '__main__':
    test = unittest.main()
