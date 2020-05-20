"""Tests for lazy reorder."""
from __future__ import division
from __future__ import print_function

import os
import math
import time
import random
import unittest
from functools import reduce
import copy
import sys
import torch
import _torch_ipex as ipex
ipex._initialize_aten_bindings()
import intel_pytorch_extension

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
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
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

    def test_Conv2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        ipex.enable_auto_dnnl()
        with torch.backends.mkldnn.flags(enabled=False):
            input = torch.rand((1, 1, 7, 7))
            for bias in [True, False]:
                input_cpu = input.clone().requires_grad_()
                input_dpcpp = input.clone().to(device=device).requires_grad_()
                conv_cpu = torch.nn.Conv2d(1, 1, (3, 3), bias=bias)
                conv_dpcpp = copy.deepcopy(conv_cpu).to(device=device)
                out_cpu = conv_cpu(input_cpu).sum()
                out_dpcpp = conv_dpcpp(input_dpcpp).sum()
                out_cpu.backward()
                out_dpcpp.backward()

                self.assertEqual(input_cpu.grad, input_dpcpp.grad)

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
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_cpu = self._seq_conf('cpu', rand_seed)

        ipex.enable_auto_dnnl()
        res_dpcpp = self._seq_conf(device, rand_seed)
        self.assertEqual(res_cpu, res_dpcpp.to('cpu'))

class TestBinaryOp(TestCase):
    def test_add(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
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

    def _test_add_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((8, 8)).to(device=device)
        a1 = a[0:2, :]
        a2 = a[4:6, :]
        self.assertEqual(a1.is_contiguous(), True)
        self.assertEqual(a2.is_contiguous(), True)
        a1 += a2
        return a1

    def test_add_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_dcpp_dnnl = self._test_add_("dpcpp:0", rand_seed)

        ipex.disable_auto_dnnl()
        res_dcpp_cpu = self._test_add_("dpcpp:0", rand_seed)

        res_cpu = self._test_add_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

    def test_add_scalar(self):
        ipex.enable_auto_dnnl()
        a = torch.rand((8, 8)).to(device=device)
        a += 2

    def test_mul(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        alpha = torch.randn(1, dtype=torch.float32).item()

        x_cpu = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y_cpu = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        y_dpcpp = y_cpu.to(device=device)

        # mul
        self.assertEqual(
            x_cpu * y_cpu,
            x_dpcpp * y_dpcpp)

        self.assertEqual(
            torch.mul(x_cpu, y_cpu),
            torch.mul(x_dpcpp, y_dpcpp))

        # mul_out
        out_cpu = x_cpu.clone()
        out_dpcpp = out_cpu.to(device=device)
        torch.mul(x_cpu, y_cpu, out=out_cpu)
        torch.mul(x_dpcpp, y_dpcpp, out=out_dpcpp)
        self.assertEqual(out_cpu, out_dpcpp)

    def _test_mul_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((20, 20)).to(device=device)
        b = torch.rand((20, 20)).to(device=device)
        a.mul_(b)
        return a

    def test_mul_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        a1 = self._test_mul_(device, rand_seed)
        a2 = self._test_mul_('cpu', rand_seed)
        self.assertEqual(a2, a1.to('cpu'))

class TestRelu(TestCase):
    def _test_relu_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((30, 30)).to(device=device)
        a.relu_()
        return a

    def test_relu_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        a1 = self._test_relu_(device, rand_seed)
        a2 = self._test_relu_('cpu', rand_seed)
        self.assertEqual(a2, a1.to('cpu'))

    def test_relu(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn((4, 5), dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        self.assertEqual(torch.relu(x_cpu), torch.relu(x_dpcpp))

    def test_relu_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()
        y_cpu = torch.relu(x_cpu).sum()
        y_dpcpp = torch.relu(x_dpcpp).sum()
        y_cpu.backward()
        y_dpcpp.backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad)

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
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_dcpp_dnnl, input_dpcpp_dnnl, _ = self._test_conv_add_relu_("dpcpp:0", rand_seed)

        ipex.disable_auto_dnnl()
        res_dcpp_cpu, input_dpcpp_cpu, _ = self._test_conv_add_relu_("dpcpp:0", rand_seed)

        res_cpu, input_cpu, _ = self._test_conv_add_relu_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

        ipex.enable_auto_dnnl()
        res_dcpp_dnnl.sum()#.backward()
        res_dcpp_cpu.sum()#.backward()
        res_cpu.sum()#.backward()

        #self.assertEqual(input_dpcpp_dnnl.grad.to('cpu'), input_cpu.grad, prec=0.0)
        #self.assertEqual(input_dpcpp_cpu.grad.to('cpu'), input_cpu.grad, prec=0.0)

class TestLinearAlgebraOps(TestCase):
    def test_mm(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
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
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
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
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        for i in range(8, 12, 2):
            for j in range(8, 12, 2):
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
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        for i in range(8, 12, 2):
            for j in range(8, 12, 2):
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
                self.assertEqual(addbmm_cpu, addbmm_dpcpp, 1e-4)
                y_cpu = torch.randn(M, O, dtype=torch.float32)
                y_dpcpp = y_cpu.to(device=device)
                torch.addbmm(res_cpu, b1_cpu, b2_cpu, beta=beta, alpha=alpha, out=y_cpu)
                torch.addbmm(res_dpcpp, b1_dpcpp, b2_dpcpp, beta=beta, alpha=alpha, out=y_dpcpp)
                self.assertEqual(y_cpu, y_dpcpp, 1e-4)

    def test_baddbmm(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        for i in range(8, 12, 2):
            for j in range(8, 12, 2):
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
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        x_dpcpp = x.to(device=device)

        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias)
            linear_dpcpp = copy.deepcopy(linear).to(device=device)
            self.assertEqual(linear(x), linear_dpcpp(x_dpcpp))

    # we should first expose aten::linear, depend on https://github.com/pytorch/pytorch/pull/20039
    def test_linear_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        for bias in [True, False]:
            x1 = x.clone().requires_grad_()
            x2 = x.clone().to(device=device).requires_grad_()
            linear = torch.nn.Linear(in_features, out_features, bias=bias)
            linear_dpcpp =copy.deepcopy(linear).to(device=device)
            y1 = linear(x1).sum()
            y2 = linear_dpcpp(x2).sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad)

class TestPool(TestCase):
    def test_avg_pool2d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x_cpu = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)

        for count_include_pad in [True, False]:
            avg_pool2d = torch.nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(avg_pool2d(x_cpu), avg_pool2d(x_dpcpp))

    def test_avg_pool3d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x_cpu = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)

        for count_include_pad in [True, False]:
            avg_pool3d = torch.nn.AvgPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(avg_pool3d(x_cpu), avg_pool3d(x_dpcpp))

    def test_avg_pool2d_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(10, 3, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            x_cpu = x.clone().requires_grad_()
            x_dpcpp = x.clone().to(device=device).requires_grad_()
            avg_pool2d = torch.nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            y_cpu = avg_pool2d(x_cpu).sum()
            y_dpcpp = avg_pool2d(x_dpcpp).sum()
            y_cpu.backward()
            y_dpcpp.backward()
            self.assertEqual(x_cpu.grad, x_dpcpp.grad)

    def test_avg_pool3d_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(10, 3, 64, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            x_cpu = x.clone().requires_grad_()
            x_dpcpp = x.clone().to(device=device).requires_grad_()
            avg_pool3d = torch.nn.AvgPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            y_cpu = avg_pool3d(x_cpu).sum()
            y_dpcpp = avg_pool3d(x_dpcpp).sum()
            y_cpu.backward()
            y_dpcpp.backward()
            self.assertEqual(x_cpu.grad, x_dpcpp.grad)

    def test_adaptive_avg_pool2d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x_cpu = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100
        x_dpcpp = x_cpu.to(device=device)

        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        self.assertEqual(
            adaptive_avg_pool2d(x_cpu),
            adaptive_avg_pool2d(x_dpcpp))

    def test_adaptive_avg_pool2d_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(10, 3, 224, 224, dtype=torch.float32) * 100

        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        y_cpu = adaptive_avg_pool2d(x_cpu).sum()
        y_dpcpp = adaptive_avg_pool2d(x_dpcpp).sum()
        y_cpu.backward()
        y_dpcpp.backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad)

    def test_max_pool2d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                x_cpu = torch.randn(N, C, H, W, dtype=torch.float32) * 10
                x_dpcpp = x_cpu.to(device=device)

                for ceil_mode in [False, True]:
                    max_pool2d = torch.nn.MaxPool2d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    self.assertEqual(max_pool2d(x_cpu), max_pool2d(x_dpcpp))

    def test_max_pool3d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
                x_cpu = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
                x_dpcpp = x_cpu.to(device=device)

                for ceil_mode in [False, True]:
                    max_pool3d = torch.nn.MaxPool3d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    self.assertEqual(max_pool3d(x_cpu), max_pool3d(x_dpcpp))

    def test_max_pool2d_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(10, 3, 64, 64, dtype=torch.float32) * 10
        for ceil_mode in [True]:
            max_pool2d = torch.nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                ceil_mode=ceil_mode)

            x1 = x.clone().requires_grad_()
            x2 = x.clone().to(device=device).requires_grad_()

            y1 = max_pool2d(x1).sum()
            y2 = max_pool2d(x2).sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad)

    def test_max_pool3d_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
                x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
                x1 = x.clone().requires_grad_()
                x2 = x.clone().to(device=device).requires_grad_()

                for ceil_mode in [False, True]:
                    max_pool3d = torch.nn.MaxPool3d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    y1 = max_pool3d(x1).sum()
                    y2 = max_pool3d(x2).sum()
                    y1.backward()
                    y2.backward()
                    self.assertEqual(x1.grad, x2.grad)

class TestBatchNorm(TestCase):
    def test_batch_norm2d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)

        bn = torch.nn.BatchNorm2d(3)
        bn_dpcpp =copy.deepcopy(bn).to(device=device)
        self.assertEqual(bn(x_cpu), bn_dpcpp(x_dpcpp))

    def test_batch_norm3d(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(4, 3, 30, 30, 30, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)

        bn = torch.nn.BatchNorm3d(3)
        bn_dpcpp = copy.deepcopy(bn).to(device=device)
        self.assertEqual(bn(x_cpu), bn_dpcpp(x_dpcpp))

    def test_batch_norm2d_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10
        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()

        bn = torch.nn.BatchNorm2d(3)
        bn_dpcpp = copy.deepcopy(bn).to(device=device)
        y_cpu = bn(x_cpu).sum()
        y_dpcpp = bn_dpcpp(x_dpcpp).sum()
        y_cpu.backward()
        y_dpcpp.backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad)

class TestTensorShape(TestCase):
    def test_reshape(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        self.assertEqual(torch.reshape(x_cpu, (6, 10)), torch.reshape(x_dpcpp, (6, 10)))

    def test_cat(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(4, 5, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        for dim in [0, 1]:
            self.assertEqual(torch.cat((x_cpu, x_cpu, x_cpu), dim=dim), torch.cat((x_dpcpp, x_dpcpp, x_dpcpp), dim=dim))
        #cat_out
        y_cpu = torch.randn(12, 5, dtype=torch.float32)*10
        y_dpcpp = y_cpu.to(device=device)
        torch.cat((x_cpu, x_cpu, x_cpu), dim=0, out=y_cpu),
        torch.cat((x_dpcpp, x_dpcpp, x_dpcpp), dim=0, out=y_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)
        y_cpu = torch.randn(4, 15, dtype=torch.float32)*10
        y_dpcpp = y_cpu.to(device=device)
        torch.cat((x_cpu, x_cpu, x_cpu), dim=1, out=y_cpu),
        torch.cat((x_dpcpp, x_dpcpp, x_dpcpp), dim=1, out=y_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)

    def test_cat_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()
        y_cpu = torch.cat((x_cpu, x_cpu, x_cpu)).sum()
        y_dpcpp = torch.cat((x_dpcpp, x_dpcpp, x_dpcpp)).sum()
        y_cpu.backward()
        y_dpcpp.backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad)

    def test_transpose(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        x_dpcpp = x.clone().to(device=device)
        for dim1 in range(x.ndim):
            for dim2 in range(x.ndim):
                self.assertEqual(
                    x.transpose(dim1, dim2),
                    x_dpcpp.transpose(dim1, dim2),
                )

    def test_view(self):
        ipex.enable_auto_dnnl()
        old_shape = (4, 16)
        new_shape = (1, 4, 4, 4)

        x_cpu = torch.randn(old_shape)
        x_dpcpp = x_cpu.to(device=device).clone()
        self.assertTrue(ipex.is_dil_tensor(x_dpcpp))
        self.assertEqual(ipex.get_dil_tensor_sizes(x_dpcpp), [4, 16])
        self.assertEqual(ipex.get_dil_tensor_strides(x_dpcpp), [16, 1])

        x_cpu_view = x_cpu.view(new_shape)
        self.assertEqual(x_cpu_view.size(), [1, 4, 4, 4])
        self.assertEqual(x_cpu_view.stride(), [64, 16, 4, 1])

        x_dpcpp_view = x_dpcpp.view(new_shape)
        self.assertTrue(ipex.is_dil_tensor(x_dpcpp_view))
        
        y = torch.randn(new_shape)
        out_cpu = x_cpu_view * y
        # test if the shape of x_dpcpp_view is compatible with y
        out_dpcpp = x_dpcpp_view * y
        self.assertTrue(ipex.is_dil_tensor(out_dpcpp))
        self.assertEqual(ipex.get_dil_tensor_sizes(out_dpcpp), [1, 4, 4, 4])
        self.assertEqual(ipex.get_dil_tensor_strides(out_dpcpp), [64, 16, 4, 1])
        self.assertEqual(out_cpu, out_dpcpp)

        # test if metadata of x_dpcpp has not been altered
        y = torch.randn(old_shape)
        out_cpu = x_cpu * y
        out_dpcpp = x_dpcpp * y
        self.assertEqual(out_cpu, out_dpcpp)


class TestSoftMax(TestCase):
    def test_softmax(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        for dim in range(x_cpu.ndim):
            softmax = torch.nn.Softmax(dim=dim)
            self.assertEqual(softmax(x_cpu), softmax(x_dpcpp))

    def test_softmax_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        for dim in range(x.ndim):
            x_cpu = x.clone().requires_grad_()
            x_dpcpp = x.clone().to(device=device).requires_grad_()
            softmax = torch.nn.Softmax(dim=dim)
            y_cpu = softmax(x_cpu).sum()
            y_dpcpp = softmax(x_dpcpp).sum()
            y_cpu.backward()
            y_dpcpp.backward()
            self.assertEqual(x_cpu.grad, x_dpcpp.grad)

class TestSigmoid(TestCase):
    def test_sigmoid(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(4, 5, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        self.assertEqual(torch.sigmoid(x_cpu), torch.sigmoid(x_dpcpp))
        # inplace
        torch.sigmoid_(x_cpu)
        torch.sigmoid_(x_dpcpp)
        self.assertEqual(x_cpu, x_dpcpp)

    def test_sigmoid_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()
        y_cpu = torch.sigmoid(x_cpu)
        y_dpcpp = torch.sigmoid(x_dpcpp)
        self.assertEqual(y_cpu, y_dpcpp)

        y_cpu.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad)

class TestDropout(TestCase):
    def test_dropout(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        p = 0.2
        input = torch.randn(1000, dtype=torch.float32)
        input = input.fill_(1 - p)
        module = torch.nn.Dropout(p)
        input_dpcpp = input.clone().to(device=device).requires_grad_()
        output_dpcpp = module(input_dpcpp)
        self.assertLess(abs(output_dpcpp.data.mean() - (1 - p)), 0.05)
        output_dpcpp.backward(input_dpcpp)
        self.assertLess(abs(input_dpcpp.grad.data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = torch.nn.Dropout(p, inplace).eval()
            self.assertEqual(input_dpcpp, module(input_dpcpp))

        # Check that these don't raise errors
        module.__repr__()
        str(module)

class TestSplit(TestCase):
    def test_split(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(5, 5, dtype=torch.float32) * 10
        x_dpcpp = x.clone().to(device=device)
        for dim in [0, 1]:
            self.assertEqual(
                torch.split(x, (2,3), dim=dim)[0],
                torch.split(x_dpcpp, (2,3), dim=dim)[0],
            )
            self.assertEqual(
                torch.split(x, (2,3), dim=dim)[1],
                torch.split(x_dpcpp, (2,3), dim=dim)[1],
            )
            self.assertEqual(
                torch.split(x, 3, dim=dim)[0],
                torch.split(x_dpcpp, 3, dim=dim)[0],
            )
            self.assertEqual(
                torch.split(x, 3, dim=dim)[1],
                torch.split(x_dpcpp, 3, dim=dim)[1],
            )
            self.assertEqual(
                torch.split(x, 2, dim=dim)[0],
                torch.split(x_dpcpp, 2, dim=dim)[0],
            )
            self.assertEqual(
                torch.split(x, 2, dim=dim)[1],
                torch.split(x_dpcpp, 2, dim=dim)[1],
            )
            self.assertEqual(
                torch.split(x, 2, dim=dim)[2],
                torch.split(x_dpcpp, 2, dim=dim)[2],
            )

    def test_split_backward(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(5, 5, dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to(device=device).requires_grad_()
        for dim in [0, 1]:
            y1 = torch.split(x1, (2,3), dim=dim)[0].sum() \
                    + torch.split(x1, (2,3), dim=dim)[1].sum()
            y2 = torch.split(x2, (2,3), dim=dim)[0].sum() \
                    + torch.split(x2, (2,3), dim=dim)[1].sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad)
            y1 = torch.split(x1, 3, dim=dim)[0].sum() \
                    + torch.split(x1, 3, dim=dim)[1].sum()
            y2 = torch.split(x2, 3, dim=dim)[0].sum() \
                    + torch.split(x2, 3, dim=dim)[1].sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad)
            y1 = torch.split(x1, 2, dim=dim)[0].sum() \
                    + torch.split(x1, 2, dim=dim)[1].sum() \
                    + torch.split(x1, 2, dim=dim)[2].sum()
            y2 = torch.split(x2, 2, dim=dim)[0].sum() \
                    + torch.split(x2, 2, dim=dim)[1].sum() \
                    + torch.split(x2, 2, dim=dim)[2].sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad)

if __name__ == '__main__':
    test = unittest.main()
