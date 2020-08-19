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

class TestDeconv(TestCase):
    def test_Deconv2d_with_cpu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        _deconv = torch.nn.ConvTranspose2d(2, 3, (3, 3))

        bn_man_bf16 =copy.deepcopy(_deconv).to(device=device).to(torch.bfloat16)
        bn_auto_mix =copy.deepcopy(_deconv).to(device=device)

        _in_cpu = torch.rand((1, 2, 7, 7))
        in_auto_mix = _in_cpu.to(device=device)
        in_man_bf16 = in_auto_mix.to(torch.bfloat16)

        res_cpu_fp32 = _deconv(_in_cpu)

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

class TestShape(TestCase):
    def _check_tensor_shape(self, t1, t2):
        self.assertEqual(t1.size(), t2.size())
        self.assertEqual(t1.stride(), t2.stride())
        self.assertEqual(t1.storage_offset(), t2.storage_offset())

    def test_slice(self):
        with AutoDNNL(True), AutoMixPrecision(True):
            x_cpu = torch.rand(10, 10, 10)
            x_cpu_slice = x_cpu[3:7, 3:7, 5]

            x_dpcpp = x_cpu.to(device=device)
            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_dpcpp))

            # the storage should be converted to bf16 on slicing
            x_dpcpp_slice = x_dpcpp[3:7, 3:7, 5]
            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_dpcpp))
            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_dpcpp_slice))

            # check shape info
            self._check_tensor_shape(x_cpu, x_dpcpp)
            self._check_tensor_shape(x_cpu_slice, x_dpcpp_slice)

            # simple binary op
            y_cpu = x_cpu_slice * x_cpu_slice
            y_dpcpp = x_dpcpp_slice * x_dpcpp_slice
            self.assertEqual(y_cpu, y_dpcpp, 0.01)

            # check sliced data. This should convert the storage back to fp32
            self.assertEqual(x_cpu_slice, x_dpcpp_slice, 0.01)
            self.assertEqual(x_cpu, x_dpcpp, 0.01)
            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_dpcpp))
            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_dpcpp_slice))

            # check shape info
            self._check_tensor_shape(x_cpu, x_dpcpp)
            self._check_tensor_shape(x_cpu_slice, x_dpcpp_slice)

    def test_cat_slice(self):
        with AutoDNNL(True), AutoMixPrecision(True):
            x_cpu = torch.rand(10)
            y_cpu = torch.cat([x_cpu, x_cpu, x_cpu])

            x_dpcpp = x_cpu.to(device=device)
            y_dpcpp = torch.cat([x_dpcpp, x_dpcpp, x_dpcpp])

            res_cpu = y_cpu[0:10] * y_cpu[10:20] * y_cpu[20:30]
            res_dpcpp = y_dpcpp[0:10] * y_dpcpp[10:20] * y_dpcpp[20:30]
            self.assertEqual(res_cpu, res_dpcpp, 0.01)

    def test_sliced_binary(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        with AutoDNNL(True), AutoMixPrecision(True):
            x_cpu = torch.rand(10, 10, 10)
            x_cpu_slice = x_cpu[3:7, 3:7, 5]

            x_dpcpp = x_cpu.to(device=device)
            x_dpcpp_slice = x_dpcpp[3:7, 3:7, 5]

            # test mul
            y_cpu = x_cpu_slice * x_cpu_slice
            y_dpcpp = x_dpcpp_slice * x_dpcpp_slice
            self._check_tensor_shape(y_cpu, y_dpcpp)
            self.assertEqual(y_cpu, y_dpcpp, 0.01)

            # test sum
            y_cpu = x_cpu_slice + x_cpu_slice
            y_dpcpp = x_dpcpp_slice + x_dpcpp_slice
            self._check_tensor_shape(y_cpu, y_dpcpp)
            self.assertEqual(y_cpu, y_dpcpp, 0.01)

    def test_extract_sliced(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        with AutoDNNL(True), AutoMixPrecision(True):
            x_cpu = torch.rand(10, 10, 10)
            x_cpu_slice = x_cpu[3:7, 3:7, 5]

            x_dpcpp = x_cpu.to(device=device)
            x_dpcpp_slice = x_dpcpp[3:7, 3:7, 5]

            x_cpu_slice_clone = x_cpu_slice.clone()
            x_dpcpp_slice_clone = x_dpcpp_slice.clone()
            self._check_tensor_shape(x_cpu_slice_clone, x_dpcpp_slice_clone)
            self.assertEqual(x_cpu_slice_clone, x_dpcpp_slice_clone, 0.01)


    # def test_sliced_eltwise(self):
    #     rand_seed = int(get_rand_seed())
    #     print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
    #     torch.manual_seed(rand_seed)

    #     with AutoDNNL(True), AutoMixPrecision(True):
    #         x_cpu = torch.rand(10, 10, 10)
    #         x_cpu_slice = x_cpu[3:7, 3:7, 5]

    #         x_dpcpp = x_cpu.to(device=device)
    #         x_dpcpp_slice = x_dpcpp[3:7, 3:7, 5]

    #         y_cpu = F.relu(x_cpu_slice)
    #         y_dpcpp = F.relu(x_dpcpp_slice)
    #         self._check_tensor_shape(y_cpu, y_dpcpp)
    #         self.assertEqual(y_cpu, y_dpcpp, 0.01)

    def test_linear_with_sliced_bias(self):
        bias = torch.rand(30)
        x_cpu = torch.rand(20, 30)
        w_cpu = torch.rand(10, 30)
        b_cpu = torch.rand(30)
        y_cpu = F.linear(x_cpu, w_cpu, b_cpu[10:20])

        x_dpcpp = x_cpu.to(device=device)
        w_dpcpp = w_cpu.to(device=device)
        b_dpcpp = b_cpu.to(device=device)
        with AutoDNNL(True), AutoMixPrecision(True):
            y_dpcpp = F.linear(x_dpcpp, w_dpcpp, b_dpcpp[10:20])

        self.assertEqual(y_cpu, y_dpcpp, 0.1)

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

class ConvRelu(nn.Module):
    def __init__(self):
        super(ConvRelu, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class TestSave(TestCase):
    def test_save_and_load(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        input = torch.randn(20, 16, 50, 100)
        input_dpcpp = input.clone().to(device=device)
        model = ConvRelu()
        model_dpcpp = copy.deepcopy(model).to(device=device)
        model_bf16 = copy.deepcopy(model).to(device=device).to(torch.bfloat16)

        #test save and load model
        torch.save(model.state_dict(), 'model.pth')
        torch.save(model_bf16.state_dict(), 'model_dpcpp.pth')
        state_dict1 = torch.load('model.pth')
        state_dict2 = torch.load('model_dpcpp.pth')
        model1 = ConvRelu()
        model2 = ConvRelu()
        model1.load_state_dict(state_dict1)
        model2.load_state_dict(state_dict2)
        self.assertEqual(model1(input), model2(input), 0.01)

        #test save and load tensor
        x = torch.tensor([0, 1, 2, 3, 4])
        x_dpcpp = x.clone().to(device=device).to(torch.bfloat16)
        self.assertEqual(x_dpcpp.dtype, torch.bfloat16)
        torch.save(x, 'tensor.pt')
        torch.save(x_dpcpp, 'tensor_dpcpp.pt')
        self.assertEqual(torch.load('tensor.pt'), torch.load('tensor_dpcpp.pt'))

        with AutoDNNL(True), AutoMixPrecision(True):
            output_dpcpp = model_dpcpp(input_dpcpp)
            torch.save(output_dpcpp.clone().to('cpu'), 'tensor.pt')
            self.assertTrue(ipex.core.is_bf16_dil_tensor(output_dpcpp))
            torch.save(output_dpcpp, 'tensor_dpcpp.pt')
            self.assertEqual(torch.load('tensor.pt'), torch.load('tensor_dpcpp.pt'))


if __name__ == '__main__':
    test = unittest.main()
