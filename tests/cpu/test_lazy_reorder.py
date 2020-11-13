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
import itertools
import torch
import intel_pytorch_extension as ipex

from common_ipex_conf import AutoMixPrecision, AutoDNNL

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

device = ipex.DEVICE

def convert_blocked(t):
    assert t.dim() == 4, "only support converting 4d tensor"
    c = t.size(1)
    t = t.clone().to(device)
    return F.conv2d(t, torch.ones(c, 1, 1, 1).to(device), groups=c)

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

        ipex.core.enable_auto_dnnl()
        out_dpcpp = conv_dpcpp(input_dpcpp)

        ipex.core.disable_auto_dnnl()
        out_dpcpp_cpu = out_dpcpp.to('cpu')
        out_cpu = conv_cpu(input_cpu)
        self.assertEqual(out_dpcpp.size(), out_cpu.size())
        self.assertEqual(out_cpu, out_dpcpp_cpu)

    def test_Conv2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        ipex.core.enable_auto_dnnl()
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
        ipex.core.disable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_cpu = self._seq_conf('cpu', rand_seed)

        ipex.core.enable_auto_dnnl()
        res_dpcpp = self._seq_conf(device, rand_seed)
        self.assertEqual(res_cpu, res_dpcpp.to('cpu'))

class TestDeconv(TestCase):
    def _deconv_params_list(self):
        params_dict = {
            "input_height": [8],
            "input_width": [8],
            "input_depth": [8],
            "input_channel_per_group": [10],
            "output_channel_per_group": [10],
            "kernel_size": [3, 4],
            "bias": [False, True],
            "stride": [2], # [1, 2]
            "padding": [1, 2],
            "output_padding": [2],
            "groups": [1, 2],
            "dilation": [1, 3, 4],
        }

        params_list = []

        for key, value in params_dict.items():
            params_list.append(value)
        return params_list

    def _test_deconv(self, dims):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        params_list = self._deconv_params_list()

        for input_width, input_height, input_depth, input_channel_per_group, output_channel_per_group, kernel_size, bias, stride, padding, output_padding, groups, dilation in itertools.product(*params_list):
            if (output_padding < stride or output_padding < dilation) \
                    and ((input_height - 1) * stride - 2 * padding + dilation * (kernel_size -1 ) + output_padding + 1 > 0) \
                    and ((input_width - 1) * stride - 2 * padding + dilation * (kernel_size -1 ) + output_padding + 1 > 0) \
                    and ((input_depth - 1) * stride - 2 * padding + dilation * (kernel_size -1 ) + output_padding + 1 > 0):
                
                # mkldnn does not support the case where: 
                # padding - output_padding + stride <= 0
                # while PyTorch supports this case, will fallback in this case
                # input_width = 8
                # input_height = 8
                # input_channel_per_group = 10
                # output_channel_per_group = 10
                # kernel_size = 4
                # bias = False
                # stride = 1
                # padding = 1
                # output_padding = 2
                # groups = 1
                # dilation = 3

                ic = input_channel_per_group * groups
                oc = output_channel_per_group * groups

                if dims == 2:
                    module = torch.nn.ConvTranspose2d(ic, oc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
                    x = torch.rand((2, ic, input_height, input_width))
                elif dims == 3:
                    module = torch.nn.ConvTranspose3d(ic, oc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
                    x = torch.rand((2, ic, input_depth, input_height, input_width))

                module_dpcpp = copy.deepcopy(module).to(device=device)
                module_fallback = copy.deepcopy(module).to(device=device)
                module_fallback_ = copy.deepcopy(module).to(device=device)

                x_aten = x.clone().requires_grad_()
                x_dpcpp = x.clone().to(device=device).requires_grad_()
                x_fallback = x.clone().to(device=device).requires_grad_()
                x_fallback_ = x.clone().to(device=device).requires_grad_()

                y_aten = module(x_aten)
                y_aten.sum().backward()

                # test dnnl
                with AutoDNNL(True):
                    y_dpcpp = module_dpcpp(x_dpcpp)
                    y_dpcpp.sum().backward()
                    
                    self.assertEqual(
                        y_aten, y_dpcpp)
                    self.assertEqual(
                        module.weight.grad, module_dpcpp.weight.grad, 1e-3)
                    self.assertEqual(x_aten.grad, x_dpcpp.grad)
                    if bias:
                        self.assertEqual(module.bias.grad, module_dpcpp.bias.grad)

                # test fallback
                with AutoDNNL(False):
                    y_fallback = module_fallback(x_fallback)
                    y_fallback.sum().backward()

                    self.assertEqual(
                        y_aten, y_fallback)
                    self.assertEqual(
                        module.weight.grad, module_fallback.weight.grad)
                    self.assertEqual(x_aten.grad, x_fallback.grad)
                    if bias:
                        self.assertEqual(module.bias.grad, module_fallback.bias.grad)

                # test fw: dnnl, bw: cpu
                with AutoDNNL(True):
                    y_fallback_ = module_fallback_(x_fallback_)
                with AutoDNNL(False):
                    y_fallback_.sum().backward()

                self.assertEqual(
                    y_aten, y_fallback_)
                self.assertEqual(
                    module.weight.grad, module_fallback_.weight.grad)
                self.assertEqual(x_aten.grad, x_fallback_.grad)
                if bias:
                    self.assertEqual(module.bias.grad, module_fallback_.bias.grad)
    
    def test_deconv2d(self):
        self._test_deconv(dims=2)
    
    def test_deconv3d(self):
        self._test_deconv(dims=3)

    def _seq_conf(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        deconv_dpcpp1 = torch.nn.ConvTranspose2d(2, 3, (7, 7)).to(device=device)
        deconv_dpcpp2 = torch.nn.ConvTranspose2d(3, 4, (5, 5)).to(device=device)
        deconv_dpcpp3 = torch.nn.ConvTranspose2d(4, 5, (3, 3)).to(device=device)
        input_cpu = torch.rand((1, 2, 10, 10))
        input_dpcpp = input_cpu.to(device=device)

        out_dpcpp1 = deconv_dpcpp1(input_dpcpp)
        out_dpcpp2 = deconv_dpcpp2(out_dpcpp1)
        out_dpcpp3 = deconv_dpcpp3(out_dpcpp2)
        return out_dpcpp3

    def test_seq_deconv(self):
        ipex.core.disable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_cpu = self._seq_conf('cpu', rand_seed)

        ipex.core.enable_auto_dnnl()
        res_dpcpp = self._seq_conf(device, rand_seed)
        self.assertEqual(res_cpu, res_dpcpp.to('cpu'))

class TestBinaryOp(TestCase):
    def test_add(self):
        # rand_seed = 1599794793172034560: AssertionError: tensor(1.5259e-05) not less than or equal to 1e-05
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_dcpp_dnnl = self._test_add_(device, rand_seed)

        ipex.core.disable_auto_dnnl()
        res_dcpp_cpu = self._test_add_(device, rand_seed)

        res_cpu = self._test_add_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

    def test_add_scalar(self):
        ipex.core.enable_auto_dnnl()
        a = torch.rand((8, 8)).to(device=device)
        a += 2

    def test_mul(self):
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        a1 = self._test_mul_(device, rand_seed)
        a2 = self._test_mul_('cpu', rand_seed)
        self.assertEqual(a2, a1.to('cpu'))

    def test_binary_propagate_group(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        ipex.core.enable_auto_dnnl()

        input = torch.rand((1, 64, 7, 7))

        input_cpu = input.clone().requires_grad_()
        input_dpcpp = input.clone().to(device=device).requires_grad_()
        conv_cpu = torch.nn.Conv2d(64, 64, (3, 3), groups=8)
        conv_dpcpp = copy.deepcopy(conv_cpu).to(device=device)
        conv_cpu(input_cpu)
        conv_dpcpp(input_dpcpp)

        y_cpu = conv_cpu.weight.add(conv_cpu.weight)
        y_dpcpp = conv_dpcpp.weight.add(conv_dpcpp.weight)
        self.assertEqual(y_cpu, y_dpcpp)

        y_cpu = conv_cpu.weight.mul(conv_cpu.weight)
        y_dpcpp = conv_dpcpp.weight.mul(conv_dpcpp.weight)
        self.assertEqual(y_cpu, y_dpcpp)

    def test_mixed_format(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        shape = (2, 3, 4, 5)

        for fname in ['add', 'mul']:

            x_cpu = torch.ones(shape) * 5
            y_cpu = torch.ones(shape) * 4

            # block tensor is a dpcpp tensor
            x_plain = x_cpu.clone().to(device)
            y_plain = y_cpu.clone().to(device)
            x_block = convert_blocked(x_cpu.clone())
            y_block = convert_blocked(y_cpu.clone())

            fn = getattr(torch, fname)
            ref = fn(x_cpu, y_cpu)

            # test add, mul
            def test_outplace(a, b):
                a = a.clone()
                b = b.clone()
                self.assertEqual(fn(a, b), ref)

            test_outplace(x_plain, y_plain)
            test_outplace(x_plain, y_block)
            test_outplace(y_block, x_plain)
            test_outplace(x_block, y_block)

            # test add_out, mul_out
            def test_out(a, b, o):
                a = a.clone()
                b = b.clone()
                o = o.clone()
                y = fn(a, b, out=o)
                self.assertEqual(y, ref)
                self.assertEqual(o, ref)

            out = torch.ones(shape).to(device)
            test_out(x_plain, y_plain, out)
            test_out(x_plain, y_block, out)
            test_out(y_block, x_plain, out)
            test_out(x_block, y_block, out)
            out = torch.ones(1).to(device)
            test_out(x_plain, y_plain, out)
            test_out(x_plain, y_block, out)
            test_out(y_block, x_plain, out)
            test_out(x_block, y_block, out)

            # test add_, mul_
            def test_inplace(a, b):
                a = a.clone()
                b = b.clone()
                y = getattr(a, fname + '_')(b)
                self.assertEqual(a, ref)
                self.assertEqual(y, ref)

            test_inplace(x_plain, y_plain)
            test_inplace(x_plain, y_block)
            test_inplace(y_block, x_plain)
            test_inplace(x_block, y_block)

            # test broadcast
            scalar = torch.ones(1).to(device)
            self.assertEqual(fn(x_plain, scalar), fn(x_cpu, scalar))
            self.assertEqual(fn(scalar, x_plain), fn(scalar, x_cpu))


class TestRelu(TestCase):
    def _test_relu_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        a = torch.rand((30, 30)).to(device=device)
        a.relu_()
        return a

    def test_relu_(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        a1 = self._test_relu_(device, rand_seed)
        a2 = self._test_relu_('cpu', rand_seed)
        self.assertEqual(a2, a1.to('cpu'))

    def test_relu(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn((4, 5), dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        self.assertEqual(torch.relu(x_cpu), torch.relu(x_dpcpp))

    def test_relu_backward(self):
        ipex.core.enable_auto_dnnl()
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

class TestGelu(TestCase):
    def test_gelu(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn((4, 5), dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)
        self.assertEqual(F.gelu(x_cpu), F.gelu(x_dpcpp), 0.001)

    def test_gelu_backward(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()
        y_cpu = F.gelu(x_cpu).sum()
        y_dpcpp = F.gelu(x_dpcpp).sum()
        y_cpu.backward()
        y_dpcpp.backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad, 0.001)

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

    def _test_conv_relu_(self, device, rand_seed):
        ipex.core.enable_auto_dnnl()
        torch.manual_seed(rand_seed)
        conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=device)
        conv_op_input = torch.rand((1, 1, 10, 10)).to(device=device)
        conv_op_output = conv_op(conv_op_input)
        conv_op_output.relu_()
        return conv_op_output

    def test_conv_relu_(self):
        rand_seed = int(get_rand_seed())
        res_dcpp_dnnl = self._test_conv_relu_(device, rand_seed)
        self.assertTrue(ipex.core.is_dil_tensor(res_dcpp_dnnl))
        res_cpu = self._test_conv_relu_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

    def test_conv_add_relu_(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        res_dcpp_dnnl, input_dpcpp_dnnl, _ = self._test_conv_add_relu_(device, rand_seed)

        ipex.core.disable_auto_dnnl()
        res_dcpp_cpu, input_dpcpp_cpu, _ = self._test_conv_add_relu_(device, rand_seed)

        res_cpu, input_cpu, _ = self._test_conv_add_relu_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

        ipex.core.enable_auto_dnnl()
        res_dcpp_dnnl.sum().backward()
        res_dcpp_cpu.sum().backward()
        res_cpu.sum().backward()

        self.assertEqual(input_dpcpp_dnnl.grad.to('cpu'), input_cpu.grad, prec=0.0)
        self.assertEqual(input_dpcpp_cpu.grad.to('cpu'), input_cpu.grad, prec=0.0)

class TestLinearAlgebraOps(TestCase):
    def test_mm(self):
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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

                res_cpu.addmm_(mat1=b1_cpu, mat2=b2_cpu, alpha=alpha, beta=beta)
                res_dpcpp.addmm_(mat1=b1_cpu, mat2=b2_cpu, alpha=alpha, beta=beta)
                self.assertEqual(res_cpu, res_dpcpp)


    def test_addbmm(self):
        ipex.core.enable_auto_dnnl()
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

                res_cpu.addbmm_(b1_cpu, b2_cpu, beta=beta, alpha=alpha)
                res_dpcpp.addbmm_(b1_dpcpp, b2_dpcpp, beta=beta, alpha=alpha)
                self.assertEqual(res_cpu, res_dpcpp, 1e-4)

    def test_baddbmm(self):
        ipex.core.enable_auto_dnnl()
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
                res_cpu.baddbmm_(b1_cpu, b2_cpu, alpha=alpha, beta=beta)
                res_dpcpp.baddbmm_(b1_cpu, b2_cpu, alpha=alpha, beta=beta)
                self.assertEqual(res_cpu, res_dpcpp)

class TestLinear(TestCase):
    def test_linear(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        for bias in [True, False]:
            input_shape = [in_features]
            for input_dim in [1, 2, 3, 4]:
                if input_dim != 1:  input_shape.insert(0, torch.randint(3, 10, (1,)).item())
                x = torch.randn(input_shape, dtype=torch.float32) * 10
                x_dpcpp = x.to(device=device)
                linear = torch.nn.Linear(in_features, out_features, bias=bias)
                linear_dpcpp = copy.deepcopy(linear).to(device=device)
                self.assertEqual(linear(x), linear_dpcpp(x_dpcpp))

    # we should first expose aten::linear, depend on https://github.com/pytorch/pytorch/pull/20039
    def test_linear_backward(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        for bias in [True, False]:
            input_shape = [in_features]
            for input_dim in [1, 2, 3, 4]:
                if input_dim != 1:  input_shape.insert(0, torch.randint(3, 10, (1,)).item())
                x  = torch.randn(input_shape, dtype=torch.float32) * 10
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
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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

    def test_adaptive_avg_pool2d_not_divisible(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x_cpu = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100
        x_dpcpp = x_cpu.to(device=device)
        # test the fallback to cpu when the input size is not divisible by the output size
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(6)

        y_cpu = adaptive_avg_pool2d(x_cpu)
        y_dpcpp = adaptive_avg_pool2d(x_dpcpp)

        self.assertEqual(
            y_cpu,
            y_dpcpp)

        self.assertEqual(device, y_dpcpp.device.type)

    def test_adaptive_avg_pool2d_backward_not_divisible(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.randn(10, 3, 224, 224, dtype=torch.float32) * 100

        x_cpu = x.clone().requires_grad_()
        x_dpcpp = x.clone().to(device=device).requires_grad_()
        # test the fallback to cpu when the input size is not divisible by the output size
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(6)

        y_cpu = adaptive_avg_pool2d(x_cpu).sum()
        y_dpcpp = adaptive_avg_pool2d(x_dpcpp).sum()
        y_cpu.backward()
        y_dpcpp.backward()
        self.assertEqual(x_cpu.grad, x_dpcpp.grad)

        self.assertEqual(device, x_dpcpp.grad.device.type)
        self.assertEqual(device, y_dpcpp.device.type)

    def test_max_pool2d(self):
        ipex.core.enable_auto_dnnl()
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

    def test_max_pool2d_double(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                # test the fallback to cpu when the input is double
                x_cpu = torch.randn(N, C, H, W, dtype=torch.double) * 10
                x_dpcpp = x_cpu.to(device=device)

                for ceil_mode in [False, True]:
                    max_pool2d = torch.nn.MaxPool2d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    y_cpu = max_pool2d(x_cpu)
                    y_dpcpp = max_pool2d(x_dpcpp)
                    self.assertEqual(y_cpu, y_dpcpp)

                    self.assertEqual(device, y_dpcpp.device.type)

    def test_max_pool3d(self):
        ipex.core.enable_auto_dnnl()
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
        ipex.core.enable_auto_dnnl()
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

    def test_max_pool2d_backward_double(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        # test the fallback to cpu when the input is double
        x = torch.randn(10, 3, 64, 64, dtype=torch.double) * 10
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

            self.assertEqual(device, x2.grad.device.type)
            self.assertEqual(device, y2.device.type)

    def test_max_pool3d_backward(self):
        ipex.core.enable_auto_dnnl()
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
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x_cpu = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10
            x_dpcpp = x_cpu.to(device=device)

            bn = torch.nn.BatchNorm2d(3)
            bn_dpcpp =copy.deepcopy(bn).to(device=device)
            self.assertEqual(bn(x_cpu), bn_dpcpp(x_dpcpp))

    def test_batch_norm3d(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x_cpu = torch.randn(4, 3, 30, 30, 30, dtype=torch.float32) * 10
            x_dpcpp = x_cpu.to(device=device)

            bn = torch.nn.BatchNorm3d(3)
            bn_dpcpp = copy.deepcopy(bn).to(device=device)
            self.assertEqual(bn(x_cpu), bn_dpcpp(x_dpcpp))

    def test_batch_norm2d_backward(self):
        with AutoDNNL(True):
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

class TestLayerNorm(TestCase):
    def test_layer_norm(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            input = torch.randn(2, 5, 10, 10, dtype=torch.float32)
            input_dpcpp=input.to(device=device)
            m = torch.nn.LayerNorm([10, 10])
            m_dpcpp = copy.deepcopy(m).to(device=device)
            output = m(input)
            output_dpcpp = m_dpcpp(input_dpcpp)
            self.assertTrue(ipex.core.is_dil_tensor(output_dpcpp))
            self.assertEqual(output, output_dpcpp)

    def test_layer_norm_backward(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            input = torch.randn(2, 5, 10, 10, dtype=torch.float32)
            input_cpu = input.clone().requires_grad_()
            input_dpcpp=input.clone().to(device=device).requires_grad_()
            m = torch.nn.LayerNorm([10, 10])
            m_dpcpp = copy.deepcopy(m).to(device=device)
            y_cpu = m(input_cpu).sum()
            y_cpu.backward()
            y_dpcpp = m_dpcpp(input_dpcpp).sum()
            y_dpcpp.backward()
            self.assertEqual(input_cpu.grad, input_dpcpp.grad)

class TestTensorShape(TestCase):
    def test_reshape(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x_cpu = torch.randn(3, 4, 5, dtype=torch.float32) * 10
            x_dpcpp = x_cpu.to(device=device)
            self.assertEqual(torch.reshape(x_cpu, (6, 10)), torch.reshape(x_dpcpp, (6, 10)))

    def test_cat(self):
        with AutoDNNL(True):
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
        with AutoDNNL(True):
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
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x = torch.randn(3, 4, 5, 6, dtype=torch.float32)
            x_dpcpp = x.clone().to(device=device)
            for dim1 in range(x.ndim):
                for dim2 in range(x.ndim):
                    ref = x.transpose(dim1, dim2)
                    self.assertEqual(
                        ref, x_dpcpp.transpose(dim1, dim2))

                    x_dpcpp_blocked = convert_blocked(x_dpcpp)
                    self.assertEqual(
                        ref, x_dpcpp_blocked.transpose(dim1, dim2))

    def test_view(self):
        with AutoDNNL(True):
            old_shape = (4, 16)
            new_shape = (1, 4, 4, 4)

            x_cpu = torch.randn(old_shape)
            x_dpcpp = x_cpu.to(device=device).clone()
            self.assertTrue(ipex.core.is_dil_tensor(x_dpcpp))
            self.assertEqual(ipex.core.get_dil_tensor_sizes(x_dpcpp), [4, 16])
            self.assertEqual(ipex.core.get_dil_tensor_strides(x_dpcpp), [16, 1])

            x_cpu_view = x_cpu.view(new_shape)
            self.assertEqual(x_cpu_view.size(), [1, 4, 4, 4])
            self.assertEqual(x_cpu_view.stride(), [64, 16, 4, 1])

            x_dpcpp_view = x_dpcpp.view(new_shape)
            self.assertTrue(ipex.core.is_dil_tensor(x_dpcpp_view))

            y = torch.randn(new_shape)
            out_cpu = x_cpu_view * y
            # test if the shape of x_dpcpp_view is compatible with y
            out_dpcpp = x_dpcpp_view * y.to(device)
            self.assertTrue(ipex.core.is_dil_tensor(out_dpcpp))
            self.assertEqual(ipex.core.get_dil_tensor_sizes(out_dpcpp), [1, 4, 4, 4])
            self.assertEqual(ipex.core.get_dil_tensor_strides(out_dpcpp), [64, 16, 4, 1])
            self.assertEqual(out_cpu, out_dpcpp)

            # test if metadata of x_dpcpp has not been altered
            y = torch.randn(old_shape)
            out_cpu = x_cpu * y
            out_dpcpp = x_dpcpp * y
            self.assertEqual(out_cpu, out_dpcpp)

            with  AutoMixPrecision(True):
                # test share storage for view
                src_1 = torch.randn(5120, 1, 128, device=device)
                src_2 = torch.randn(5120, 1, 128, device=device)
                # res_bf16 will not be bf16 dil tensor, since add will only reorder the second
                # input to the data type of the first input if they are different
                res_bf16 = src_1 + src_2
                res_bf16_other = src_1 + src_2
                self.assertTrue(ipex.core.is_dil_tensor(res_bf16))
                # self.assertTrue(ipex.core.is_bf16_dil_tensor(res_bf16))
                self.assertTrue(ipex.core.get_dil_tensor_sizes(res_bf16), [5120, 1, 128])
                self.assertEqual(list(res_bf16.size()), [5120, 1, 128])
                res_fp32_view = res_bf16.view(1280, 4, 1, 128)
                self.assertTrue(ipex.core.is_dil_tensor(res_bf16))
                self.assertTrue(ipex.core.is_dil_tensor(res_fp32_view))
                # self.assertTrue(ipex.core.is_bf16_dil_tensor(res_bf16))
                # self.assertTrue(ipex.core.is_bf16_dil_tensor(res_fp32_view))
                self.assertEqual(list(res_fp32_view.size()), [1280, 4, 1, 128])
                tmp_res = res_bf16 + res_bf16_other
                # self.assertTrue(ipex.core.is_bf16_dil_tensor(res_bf16))
                # self.assertTrue(ipex.core.is_bf16_dil_tensor(res_fp32_view))
                tmp_res = res_fp32_view.index_select(0, torch.LongTensor([0, 1]))
                self.assertTrue(ipex.core.get_dil_tensor_sizes(res_fp32_view), [5120, 1, 128])
                self.assertTrue(ipex.core.get_dil_tensor_sizes(res_fp32_view), [5120, 1, 128])
                self.assertEqual(list(tmp_res.size()), [2, 4, 1, 128])

    def test_view_blocked(self):
        with AutoDNNL(True):
            old_shape = (1, 4, 4, 4)
            new_shape = (4, 16)

            x_cpu = torch.randn(old_shape)
            x_dpcpp = x_cpu.to(device=device)
            x_dpcpp = convert_blocked(x_dpcpp)

            x_cpu_view = x_cpu.view(new_shape)
            x_dpcpp_view = x_dpcpp.view(new_shape)

            x_cpu_view_clone = x_cpu_view.clone()
            x_dpcpp_view_clone = x_dpcpp_view.clone()
            self.assertEqual(x_cpu_view_clone, x_dpcpp_view_clone)

    def test_select(self):
        with AutoDNNL(True):
            x_cpu = torch.randn((2, 4, 4, 4))
            x_dpcpp = x_cpu.to(device=device)
            x_dpcpp = convert_blocked(x_dpcpp)

            x_cpu_view = x_cpu[0]
            x_dpcpp_view = x_dpcpp[0]

            x_cpu_view_clone = x_cpu_view.clone()
            x_dpcpp_view_clone = x_dpcpp_view.clone()
            self.assertEqual(x_cpu_view_clone, x_dpcpp_view_clone)

class TestSoftMax(TestCase):
    def test_softmax(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x_cpu = torch.randn(3, 4, 5, dtype=torch.float32) * 10
            x_dpcpp = x_cpu.to(device=device)
            for dim in range(x_cpu.ndim):
                softmax = torch.nn.Softmax(dim=dim)
                self.assertEqual(softmax(x_cpu), softmax(x_dpcpp))

    def test_softmax_backward(self):
        with AutoDNNL(True):
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

    def test_log_softmax(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        with AutoDNNL(True):
            x_cpu = torch.randn(3, 4, 5, dtype=torch.float32) * 10
            x_dpcpp = x_cpu.to(device=device)
            for dim in range(x_cpu.ndim):
                self.assertEqual(F.log_softmax(x_cpu, dim=dim), F.log_softmax(x_dpcpp, dim=dim))

    def test_log_softmax_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        with AutoDNNL(True):
            x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
            for dim in range(x.ndim):
                x_cpu = x.clone().requires_grad_()
                x_dpcpp = x.clone().to(device=device).requires_grad_()
                y_cpu = F.log_softmax(x_cpu, dim=dim).sum()
                y_dpcpp = F.log_softmax(x_dpcpp, dim=dim).sum()
                y_cpu.backward()
                y_dpcpp.backward()
                self.assertEqual(x_cpu.grad, x_dpcpp.grad, 1e-4)

class TestSigmoid(TestCase):
    def test_sigmoid(self):
        with AutoDNNL(True):
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
        with AutoDNNL(True):
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

class TestTanh(TestCase):
    def test_tanh(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x_cpu = torch.randn(4, 5, dtype=torch.float32) * 10
            x_dpcpp = x_cpu.to(device=device)
            self.assertEqual(torch.tanh(x_cpu), torch.tanh(x_dpcpp))
            # inplace
            torch.tanh_(x_cpu)
            torch.tanh_(x_dpcpp)
            self.assertEqual(x_cpu, x_dpcpp)

    def test_tanh_backward(self):
        with AutoDNNL(True):
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
            x = torch.randn(4, 5, dtype=torch.float32) * 10
            x_cpu = x.clone().requires_grad_()
            x_dpcpp = x.clone().to(device=device).requires_grad_()
            y_cpu = torch.tanh(x_cpu)
            y_dpcpp = torch.tanh(x_dpcpp)
            self.assertEqual(y_cpu, y_dpcpp)

            y_cpu.sum().backward()
            y_dpcpp.sum().backward()
            self.assertEqual(x_cpu.grad, x_dpcpp.grad)

class TestDropout(TestCase):
    def test_dropout(self):
        with AutoDNNL(True):
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
        with AutoDNNL(True):
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
        with AutoDNNL(True):
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

    def test_split_share_memory(self):
        with AutoDNNL(True):
            x_dpcpp = torch.FloatTensor([1, 1, 1, 1, -1, -1, -1, -1]).to(device=device)
            other = torch.FloatTensor([-1, -1, -1, -1]).to(device=device)

            x_target = torch.FloatTensor([0, 0, 0, 0, -1, -1, -1, -1]).to(device=device)

            splited_x = torch.split(x_dpcpp, 4)
            splited_x[0].add_(other)

            self.assertEqual(x_dpcpp, x_target)

class ConvRelu(nn.Module):
    def __init__(self):
        super(ConvRelu, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class TestSave(TestCase):
    def test_save_and_load_tensor(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x = torch.tensor([0, 1, 2, 3, 4])
        x_dpcpp = x.clone().to(device=device)
        torch.save(x, 'tensor.pt')
        torch.save(x_dpcpp, 'tensor_dpcpp.pt')
        self.assertEqual(torch.load('tensor.pt'), torch.load('tensor_dpcpp.pt'))

    def test_save_and_load_model(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        input = torch.randn(20, 16, 50, 100)
        model = ConvRelu()
        model_dpcpp = copy.deepcopy(model).to(device=device)
        torch.save(model.state_dict(), 'model.pth')
        torch.save(model_dpcpp.state_dict(), 'model_dpcpp.pth')
        state_dict1 = torch.load('model.pth')
        state_dict2 = torch.load('model_dpcpp.pth')
        model1 = ConvRelu()
        model2 = ConvRelu()
        model1.load_state_dict(state_dict1)
        model2.load_state_dict(state_dict2)
        self.assertEqual(model1(input), model2(input))

class TestRNN(TestCase):
    def test_lstm(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 1).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        c = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy[0], hy_dpcpp[0])
        self.assertEqual(hy[1], hy_dpcpp[1])
        y.sum().backward(retain_graph=True)
        y_dpcpp.sum().backward(retain_graph=True)
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)
        self.assertEqual(model_dpcpp.bias_ih_l0.grad.to('cpu'), model.bias_ih_l0.grad)
        self.assertEqual(model_dpcpp.bias_hh_l0.grad.to('cpu'), model.bias_hh_l0.grad)
        self.assertEqual(model_dpcpp.weight_ih_l0.grad.to('cpu'), model.weight_ih_l0.grad)
        self.assertEqual(model_dpcpp.weight_hh_l0.grad.to('cpu'), model.weight_hh_l0.grad)
        hy[0].sum().backward(retain_graph=True)
        hy_dpcpp[0].sum().backward(retain_graph=True)
        self.assertEqual(h0_dpcpp.grad.to('cpu'), h0.grad)
        hy[1].sum().backward(retain_graph=True)
        hy_dpcpp[1].sum().backward(retain_graph=True)
        self.assertEqual(c0_dpcpp.grad.to('cpu'), c0.grad)

    def test_lstm_no_bias(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 1, bias=False).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        c = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_lstm_dropout(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 1, dropout=1).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        c = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_lstm_dropout_inf(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 1, dropout=1).eval()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        c = torch.randn(1, 2, 3)
        input = x.clone()
        h0 = h.clone()
        c0 = c.clone()

        input_dpcpp = x.clone().to(device=device)
        h0_dpcpp = h.clone().to(device=device)
        c0_dpcpp = c.clone().to(device=device)
        model_dpcpp = copy.deepcopy(model).to(device=device)

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)

    def test_lstm_batch_first(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 1, batch_first=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 11, 3)
        c = torch.randn(1, 11, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_lstm_direction(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 1, bidirectional=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(2, 2, 3)
        c = torch.randn(2, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_lstm_layer(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 2).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(2, 2, 3)
        c = torch.randn(2, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_lstm_layer_direction(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.LSTM(5, 3, 2, bidirectional=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(4, 2, 3)
        c = torch.randn(4, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()
        c0 = c.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        c0_dpcpp = c.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, (h0, c0))
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, (h0_dpcpp, c0_dpcpp))
        self.assertEqual(y, y_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_rnn(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.RNN(5, 3, 1).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward(retain_graph=True)
        y_dpcpp.sum().backward(retain_graph=True)
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)
        self.assertEqual(model_dpcpp.bias_ih_l0.grad.to('cpu'), model.bias_ih_l0.grad)
        self.assertEqual(model_dpcpp.bias_hh_l0.grad.to('cpu'), model.bias_hh_l0.grad)
        self.assertEqual(model_dpcpp.weight_ih_l0.grad.to('cpu'), model.weight_ih_l0.grad)
        self.assertEqual(model_dpcpp.weight_hh_l0.grad.to('cpu'), model.weight_hh_l0.grad)
        hy.sum().backward(retain_graph=True)
        hy_dpcpp.sum().backward(retain_graph=True)
        self.assertEqual(h0_dpcpp.grad.to('cpu'), h0.grad)
    
    def test_rnn_relu(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.RNN(5, 3, 1, nonlinearity='relu').train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_rnn_no_bias(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.RNN(5, 3, 1, bias=False).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_rnn_dropout_inf(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.RNN(5, 3, 1, dropout=1).eval()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone()
        h0 = h.clone()

        input_dpcpp = x.clone().to(device=device)
        h0_dpcpp = h.clone().to(device=device)
        model_dpcpp = copy.deepcopy(model).to(device=device)

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)

    def test_rnn_batch_first(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.RNN(5, 3, 1, batch_first=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 11, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_rnn_layer_direction(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.RNN(5, 3, 2, bidirectional=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(4, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_gru(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.GRU(5, 3, 1).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward(retain_graph=True)
        y_dpcpp.sum().backward(retain_graph=True)
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)
        self.assertEqual(model_dpcpp.bias_ih_l0.grad.to('cpu'), model.bias_ih_l0.grad)
        self.assertEqual(model_dpcpp.bias_hh_l0.grad.to('cpu'), model.bias_hh_l0.grad)
        self.assertEqual(model_dpcpp.weight_ih_l0.grad.to('cpu'), model.weight_ih_l0.grad)
        self.assertEqual(model_dpcpp.weight_hh_l0.grad.to('cpu'), model.weight_hh_l0.grad)
        hy.sum().backward(retain_graph=True)
        hy_dpcpp.sum().backward(retain_graph=True)
        self.assertEqual(h0_dpcpp.grad.to('cpu'), h0.grad)

    def test_gru_no_bias(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.GRU(5, 3, 1, bias=False).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_gru_dropout_inf(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.GRU(5, 3, 1, dropout=1).eval()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 2, 3)
        input = x.clone()
        h0 = h.clone()

        input_dpcpp = x.clone().to(device=device)
        h0_dpcpp = h.clone().to(device=device)
        model_dpcpp = copy.deepcopy(model).to(device=device)

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)

    def test_gru_batch_first(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.GRU(5, 3, 1, batch_first=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(1, 11, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

    def test_gru_layer_direction(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        model = torch.nn.GRU(5, 3, 2, bidirectional=True).train()
        x = torch.randn(11, 2, 5)
        h = torch.randn(4, 2, 3)
        input = x.clone().requires_grad_()
        h0 = h.clone().requires_grad_()

        input_dpcpp = x.clone().to(device=device).requires_grad_()
        h0_dpcpp = h.clone().to(device=device).requires_grad_()
        model_dpcpp = copy.deepcopy(model).to(device=device).train()

        y, hy = model(input, h0)
        y_dpcpp, hy_dpcpp = model_dpcpp(input_dpcpp, h0_dpcpp)
        self.assertEqual(y, y_dpcpp)
        self.assertEqual(hy, hy_dpcpp)
        y.sum().backward()
        y_dpcpp.sum().backward()
        self.assertEqual(input_dpcpp.grad.to('cpu'), input.grad)

if __name__ == '__main__':
    test = unittest.main()
