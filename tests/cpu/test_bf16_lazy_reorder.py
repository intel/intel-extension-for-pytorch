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
import intel_pytorch_extension as ipex

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

from common_ipex_conf import AutoMixPrecision, AutoDNNL

def get_rand_seed():
    return int(time.time() * 1000000000)

device = ipex.DEVICE
class TestConv(TestCase):
    def test_Conv2d_with_cpu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        _conv = torch.nn.Conv2d(1, 1, (3, 3))

        bn_man_bf16 =copy.deepcopy(_conv).to(device=device).to(torch.bfloat16)
        bn_auto_mix =copy.deepcopy(_conv).to(device=device)

        _in_cpu = torch.rand((1, 1, 7, 7))
        in_auto_mix = _in_cpu.to(device=device)
        in_man_bf16 = in_auto_mix.to(torch.bfloat16)

        res_cpu_fp32 = _conv(_in_cpu)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = bn_man_bf16(in_man_bf16)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            self.assertEqual(res_cpu_fp32.bfloat16().float(), res_man_bf16, 1e-2)

            with AutoMixPrecision(True):
                self.assertEqual(in_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(in_auto_mix))
                res_auto_bf16 = bn_auto_mix(in_auto_mix)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_bf16))
                self.assertEqual(res_man_bf16.float(), res_auto_bf16.float(), 1e-2)

class TestBatchNorm(TestCase):
    def test_batch_norm2d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        x_auto_mix = torch.randn(64, 3, 35, 45, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        _bn = torch.nn.BatchNorm2d(3)
        bn_man_bf16 =copy.deepcopy(_bn).to(device=device)
        bn_auto_mix =copy.deepcopy(_bn).to(device=device)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = bn_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            with AutoMixPrecision(True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                res_auto_mix = bn_auto_mix(x_auto_mix)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix))

                self.assertEqual(res_bf16.float(), res_auto_mix)


    def test_batch_norm3d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_auto_mix = torch.randn(4, 3, 30, 30, 30, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        _bn = torch.nn.BatchNorm3d(3)
        bn_man_bf16 =copy.deepcopy(_bn).to(device=device)
        bn_auto_mix =copy.deepcopy(_bn).to(device=device)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = bn_man_bf16(x_man_bf16)
            self.assertEqual(x_man_bf16.dtype, torch.bfloat16)

            with AutoMixPrecision(True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                res_auto_mix = bn_auto_mix(x_auto_mix)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix))

                self.assertEqual(res_man_bf16.float(), res_auto_mix)

class TestRelu(TestCase):
    def test_relu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_fp32 = torch.randn((4, 5), dtype=torch.float32, device=device) * 10
        x_bf16 = x_fp32.to(torch.bfloat16)

        res_fp32 = torch.relu(x_fp32)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = torch.relu(x_bf16)
            self.assertEqual(res_fp32.bfloat16().float(), res_man_bf16)

            with AutoMixPrecision(True):
                res_auto_mix = torch.relu(x_fp32)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))

                res = (res_auto_mix - res_man_bf16.float()) / res_auto_mix
                res[torch.isnan(res)] = 0
                zero_base = torch.zeros_like(res)
                self.assertEqual(res, zero_base, 1e-2)

    def test_relu_(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        x_fp32 = torch.randn((4, 5), dtype=torch.float32, device=device) * 10
        x_bf16 = x_fp32.to(torch.bfloat16)

        with AutoDNNL(True), AutoMixPrecision(False):
            x_bf16.relu_()
            self.assertEqual(x_bf16.dtype, torch.bfloat16)

            with AutoMixPrecision(True):
                x_fp32.relu_()
                self.assertEqual(x_fp32.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_fp32))

                res = (x_fp32 - x_bf16.float()) / x_fp32
                res[torch.isnan(res)] = 0
                zero_base = torch.zeros_like(res)
                self.assertEqual(res, zero_base, 1e-2)

class TestGelu(TestCase):
    def test_gelu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_fp32 = torch.randn((4, 5), dtype=torch.float32, device=device) * 10
        x_bf16 = x_fp32.to(torch.bfloat16)

        res_fp32 = F.gelu(x_fp32)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = F.gelu(x_bf16)
            self.assertEqual(res_fp32.bfloat16().float(), res_man_bf16, 2e-2)

            with AutoMixPrecision(True):
                res_auto_mix = F.gelu(x_fp32)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))

                res = (res_auto_mix - res_man_bf16.float()) / res_auto_mix
                res[torch.isnan(res)] = 0
                zero_base = torch.zeros_like(res)
                self.assertEqual(res, zero_base)

class TestBinOPs(TestCase):
    def _gen_shapes(self):
        dims = torch.randint(1, 10, (1,))
        shape = torch.randint(1, 10, list(dims))
        return shape.tolist()

    # def test_add(self):
    #     rand_seed = int(get_rand_seed())
    #     print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    #     torch.manual_seed(1591058395950926848)

    #     shape = self._gen_shapes()
    #     x_auto_mix_a = torch.rand(shape, dtype=torch.float32, device=device)
    #     x_auto_mix_b = torch.rand(shape, dtype=torch.float32, device=device)
    #     x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
    #     x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)

    #     with AutoDNNL(True), AutoMixPrecision(False):
    #         res_man_bf16 = x_man_bf16_a + x_man_bf16_b
    #         self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

    #         with AutoMixPrecision(True):
    #             self.assertEqual(x_auto_mix_a.dtype, torch.float)
    #             self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
    #             self.assertEqual(x_auto_mix_b.dtype, torch.float)
    #             self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b))
    #             res_auto_mix = x_auto_mix_a + x_auto_mix_b
    #             self.assertEqual(res_auto_mix.dtype, torch.float)
    #             self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))

    def test_mul(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(1591058395950926848)

        shape = self._gen_shapes()
        x_auto_mix_a = torch.rand(shape, dtype=torch.float32, device=device)
        x_auto_mix_b = torch.rand(shape, dtype=torch.float32, device=device)
        x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
        x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = x_man_bf16_a * x_man_bf16_b
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

            with AutoMixPrecision(True):
                self.assertEqual(x_auto_mix_a.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                self.assertEqual(x_auto_mix_b.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b))
                res_auto_mix = x_auto_mix_a * x_auto_mix_b
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_mul_(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(1591058395950926848)

        shape = self._gen_shapes()
        x_auto_mix_a = torch.rand(shape, dtype=torch.float32, device=device)
        x_auto_mix_b = torch.rand(shape, dtype=torch.float32, device=device)
        x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
        x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = x_man_bf16_a * x_man_bf16_b
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            x_man_bf16_a *= x_man_bf16_b
            self.assertEqual(x_man_bf16_a.dtype, torch.bfloat16)
            self.assertEqual(x_man_bf16_a.float(), res_man_bf16.float())

            with AutoMixPrecision(True):
                self.assertEqual(x_auto_mix_a.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                self.assertEqual(x_auto_mix_b.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b))
                x_auto_mix_a *= x_auto_mix_b
                self.assertEqual(x_auto_mix_a.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                self.assertEqual(x_auto_mix_a, x_man_bf16_a.float())

class TestLinear(TestCase):
    def test_linear(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x_auto_mix = torch.randn(3, in_features, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias)

            linear_auto_mix = copy.deepcopy(linear).to(device=device)
            linear_man_bf16 = copy.deepcopy(linear).to(device=device).to(torch.bfloat16)

            with AutoDNNL(True), AutoMixPrecision(False):
                res_man_bf16 = linear_man_bf16(x_man_bf16)
                self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

                with AutoMixPrecision(True):
                    res_auto_mix = linear_auto_mix(x_auto_mix)
                    self.assertEqual(res_auto_mix.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                    self.assertEqual(res_auto_mix, res_man_bf16.float())

class TestPool(TestCase):
    def test_avg_pool2d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        x_auto_mix = torch.randn(N, C, 64, 64, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        for count_include_pad in [True, False]:
            avg_pool2d = torch.nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            avg_pool2d_auto_mix = copy.deepcopy(avg_pool2d).to(device=device)
            avg_pool2d_man_bf16 = copy.deepcopy(avg_pool2d).to(device=device).to(torch.bfloat16)

            with AutoDNNL(True), AutoMixPrecision(False):
                res_man_bf16 = avg_pool2d_man_bf16(x_man_bf16)
                self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

                with AutoMixPrecision(True):
                    res_auto_mix = avg_pool2d_auto_mix(x_auto_mix)
                    self.assertEqual(res_auto_mix.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                    self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_avg_pool3d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        x_auto_mix = torch.randn(N, C, 64, 64, 64, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        for count_include_pad in [True, False]:
            avg_pool3d = torch.nn.AvgPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            avg_pool3d_auto_mix = copy.deepcopy(avg_pool3d).to(device=device)
            avg_pool3d_man_bf16 = copy.deepcopy(avg_pool3d).to(device=device).to(torch.bfloat16)

            with AutoDNNL(True), AutoMixPrecision(False):
                res_man_bf16 = avg_pool3d_man_bf16(x_man_bf16)
                self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

                with AutoMixPrecision(True):
                    res_auto_mix = avg_pool3d_auto_mix(x_auto_mix)
                    self.assertEqual(res_auto_mix.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                    self.assertEqual(res_auto_mix, res_man_bf16.float())

class TestSoftMax(TestCase):
    def test_softmax(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_auto_mix = torch.randn(3, 4, 5, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        for dim in range(x_auto_mix.ndim):
            softmax = torch.nn.Softmax(dim=dim)

            softmax_auto_mix = copy.deepcopy(softmax).to(device=device)
            softmax_man_bf16 = copy.deepcopy(softmax).to(device=device).to(torch.bfloat16)

            with AutoDNNL(True), AutoMixPrecision(False):
                res_man_bf16 = softmax_man_bf16(x_man_bf16)
                self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

                with AutoMixPrecision(True):
                    res_auto_mix = softmax_auto_mix(x_auto_mix)
                    self.assertEqual(res_auto_mix.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                    self.assertEqual(res_auto_mix, res_man_bf16.float())

class TestSigmoid(TestCase):
    def test_sigmoid(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_auto_mix = torch.randn(4, 5, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = torch.sigmoid(x_man_bf16)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            with AutoMixPrecision(True):
                res_auto_mix = torch.sigmoid(x_auto_mix)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_sigmoid_(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_auto_mix = torch.randn(4, 5, dtype=torch.float32, device=device) * 10
        x_man_bf16 = x_auto_mix.to(torch.bfloat16)

        with AutoDNNL(True), AutoMixPrecision(False):
            torch.sigmoid_(x_man_bf16)
            with AutoMixPrecision(True):
                torch.sigmoid_(x_auto_mix)
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                self.assertEqual(x_auto_mix, x_man_bf16.float())

class TestLinearAlgebraOps(TestCase):
    def _gen_mm_tensor(self, seed, batches = None):
        torch.manual_seed(seed)
        M, N, O = 23, 8, 12
        if batches != None:
            x_auto_mix_a = torch.randn(batches, M, N, dtype=torch.float32, device=device)
            x_auto_mix_b = torch.randn(batches, N, O, dtype=torch.float32, device=device)
        else:
            x_auto_mix_a = torch.randn(M, N, dtype=torch.float32, device=device)
            x_auto_mix_b = torch.randn(N, O, dtype=torch.float32, device=device)
        res_auto_mix = torch.randn(M, O, dtype=torch.float32, device=device)
        x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
        x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)
        res_man_bf16 = res_auto_mix.to(torch.bfloat16)

        return x_auto_mix_a, x_auto_mix_b, res_auto_mix, x_man_bf16_a, x_man_bf16_b, res_man_bf16

    def test_mm(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_auto_mix_a, x_auto_mix_b, _, x_man_bf16_a, x_man_bf16_b, _ = self._gen_mm_tensor(rand_seed)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = torch.mm(x_man_bf16_a, x_man_bf16_b)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            with AutoMixPrecision(True):
                res_auto_mix = torch.mm(x_auto_mix_a, x_auto_mix_b)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_mm_out(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_auto_mix_a, x_auto_mix_b, res_auto_mix, x_man_bf16_a, x_man_bf16_b, res_man_bf16 = self._gen_mm_tensor(rand_seed)

        with AutoDNNL(True), AutoMixPrecision(False):
            torch.mm(x_man_bf16_a, x_man_bf16_b, out=res_man_bf16)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            with AutoMixPrecision(True):
                torch.mm(x_auto_mix_a, x_auto_mix_b, out=res_auto_mix)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_bmm(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_auto_mix_a, x_auto_mix_b, _, x_man_bf16_a, x_man_bf16_b, _ = self._gen_mm_tensor(rand_seed, batches=16)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = torch.bmm(x_man_bf16_a, x_man_bf16_b)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            with AutoMixPrecision(True):
                res_auto_mix = torch.bmm(x_auto_mix_a, x_auto_mix_b)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_bmm_out(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_auto_mix_a, x_auto_mix_b, res_auto_mix, x_man_bf16_a, x_man_bf16_b, res_man_bf16 = self._gen_mm_tensor(rand_seed, batches=16)
        with AutoDNNL(True), AutoMixPrecision(False):
            torch.bmm(x_man_bf16_a, x_man_bf16_b, out=res_man_bf16)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            with AutoMixPrecision(True):
                torch.bmm(x_auto_mix_a, x_auto_mix_b, out=res_auto_mix)
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_addmm(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        for i in range(8, 12, 2):
            for j in range(8, 12, 2):
                alpha = i / 10
                beta = j / 10

                x_auto_mix_a, x_auto_mix_b, add_auto_mix, x_man_bf16_a, x_man_bf16_b, add_man_bf16 = self._gen_mm_tensor(rand_seed)

                with AutoDNNL(True), AutoMixPrecision(False):
                    res_man_bf16 = torch.addmm(input=add_man_bf16, mat1=x_man_bf16_a, mat2=x_man_bf16_b, alpha=alpha, beta=beta)
                    self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
                    with AutoMixPrecision(True):
                        res_auto_mix = torch.addmm(input=add_auto_mix, mat1=x_auto_mix_a, mat2=x_auto_mix_b, alpha=alpha, beta=beta)
                        self.assertEqual(res_auto_mix.dtype, torch.float)
                        self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                        self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_addbmm(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        for i in range(8, 12, 2):
            for j in range(8, 12, 2):
                alpha = i / 10
                beta = j / 10
                num_batches = 10
                x_auto_mix_a, x_auto_mix_b, add_auto_mix, x_man_bf16_a, x_man_bf16_b, add_man_bf16 = self._gen_mm_tensor(rand_seed, num_batches)
                with AutoDNNL(True), AutoMixPrecision(False):
                    res_man_bf16 = torch.addbmm(add_man_bf16, x_man_bf16_a, x_man_bf16_b, beta=beta, alpha=alpha)
                    self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
                    with AutoMixPrecision(True):
                        res_auto_mix = torch.addbmm(add_auto_mix, x_auto_mix_a, x_auto_mix_b, beta=beta, alpha=alpha)
                        self.assertEqual(res_auto_mix.dtype, torch.float)
                        self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                        self.assertEqual(res_auto_mix, res_man_bf16.float())

    def test_baddbmm(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        for i in range(8, 12, 2):
            for j in range(8, 12, 2):
                alpha = i / 10
                beta = j / 10
                batches = 2

                M, N, O = 23, 8, 12
                x_auto_mix_a = torch.randn(batches, M, N, dtype=torch.float32, device=device)
                x_auto_mix_b = torch.randn(batches, N, O, dtype=torch.float32, device=device)
                add_auto_mix = torch.randn(batches, M, O, dtype=torch.float32, device=device)

                x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
                x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)
                add_man_bf16 = add_auto_mix.to(torch.bfloat16)

                with AutoDNNL(True), AutoMixPrecision(False):
                    res_man_bf16 = torch.baddbmm(add_man_bf16, x_man_bf16_a, x_man_bf16_b, beta=beta, alpha=alpha)
                    with AutoMixPrecision(True):
                        res_auto_mix = torch.baddbmm(add_auto_mix, x_auto_mix_a, x_auto_mix_b, beta=beta, alpha=alpha)
                        self.assertEqual(res_auto_mix.dtype, torch.float)
                        self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                        self.assertEqual(res_auto_mix, res_man_bf16.float())

if __name__ == '__main__':
    test = unittest.main()
