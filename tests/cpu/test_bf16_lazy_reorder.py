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

def get_rand_seed():
    return int(time.time() * 1000000000)

device = torch.device("dpcpp:0")
class TestConv(TestCase):
    def test_Conv2d_with_cpu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        _conv = torch.nn.Conv2d(1, 1, (3, 3))
        bn_man_bf16 =copy.deepcopy(_conv).to(device=device).to(torch.bfloat16)
        bn_auto_bf16 =copy.deepcopy(_conv).to(device=device)

        _in_cpu = torch.rand((1, 1, 7, 7))
        in_auto_bf16 = _in_cpu.to(device=device)
        in_man_bf16 = in_auto_bf16.to(torch.bfloat16)

        res_cpu_fp32 = _conv(_in_cpu)

        ipex.core.enable_auto_dnnl()
        res_man_bf16 = bn_man_bf16(in_man_bf16)
        self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
        self.assertEqual(res_cpu_fp32.bfloat16().float(), res_man_bf16, 1e-2)

        ipex.core.enable_mix_bf16_fp32()
        res_auto_bf16 = bn_auto_bf16(in_auto_bf16)
        self.assertEqual(res_auto_bf16.dtype, torch.bfloat16)
        self.assertEqual(res_man_bf16.float(), res_auto_bf16.float(), 1e-2)

# class TestBatchNorm(TestCase):
#     def test_batch_norm2d(self):
#         ipex.core.enable_auto_dnnl()
#         rand_seed = int(get_rand_seed())
#         rand_seed = 1
#         print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
#         torch.manual_seed(rand_seed)
#         x_fp32 = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10
#         x_bf16 = x_fp32.to(torch.bfloat16)

#         _bn = torch.nn.BatchNorm2d(3)
#         bn_fp32 =copy.deepcopy(_bn).to(device=device)
#         bn_bf16 =copy.deepcopy(_bn).to(device=device).to(torch.bfloat16)
#         res_bf16 = bn_bf16(x_bf16)
#         res_fp32 = bn_fp32(x_fp32)
#         self.assertEqual(bn_fp32.weight, bn_bf16.weight.float())
#         self.assertEqual(bn_fp32.bias, bn_bf16.bias.float())
#         self.assertEqual(res_bf16.dtype, torch.bfloat16)
#         self.assertEqual(res_fp32.dtype, torch.float)
#         self.assertEqual(res_fp32, res_bf16.float())

#     def test_batch_norm3d(self):
#         ipex.core.enable_auto_dnnl()
#         rand_seed = int(get_rand_seed())
#         rand_seed = 1
#         print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
#         torch.manual_seed(rand_seed)
#         x_fp32 = torch.randn(4, 3, 30, 30, 30, dtype=torch.float32) * 10
#         x_bf16 = x_fp32.to(torch.bfloat16)

#         _bn = torch.nn.BatchNorm3d(3)
#         bn_fp32 =copy.deepcopy(_bn).to(device=device)
#         bn_bf16 =copy.deepcopy(_bn).to(device=device).to(torch.bfloat16)
#         res_bf16 = bn_bf16(x_bf16.to(device=device))
#         res_fp32 = bn_fp32(x_fp32.to(device=device))
#         self.assertEqual(bn_fp32.weight, bn_bf16.weight.float())
#         self.assertEqual(bn_fp32.bias, bn_bf16.bias.float())
#         self.assertEqual(res_fp32, res_bf16.float())
#         self.assertEqual(res_bf16.dtype, torch.bfloat16)
#         self.assertEqual(res_fp32.dtype, torch.float)

class TestRelu(TestCase):
    def test_relu(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_fp32 = torch.randn((4, 5), dtype=torch.float32, device=device) * 10
        x_bf16 = x_fp32.to(torch.bfloat16)
        res_fp32 = torch.relu(x_fp32)
        res_man_bf16 = torch.relu(x_bf16)
        self.assertEqual(res_fp32.bfloat16().float(), res_man_bf16)

        ipex.core.enable_mix_bf16_fp32()
        res_auto_bf16 = torch.relu(x_fp32)
        self.assertEqual(res_auto_bf16.dtype, torch.bfloat16)
        res = (res_auto_bf16 - res_man_bf16) / res_auto_bf16
        res[torch.isnan(res)] = 0
        zero_base = torch.zeros_like(res)
        self.assertEqual(res, zero_base, 1e-2)

    def test_relu_(self):
        ipex.core.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        x_fp32 = torch.randn((4, 5), dtype=torch.float32, device=device) * 10

        x_bf16 = x_fp32.to(torch.bfloat16)
        x_bf16.relu_()
        self.assertEqual(x_bf16.dtype, torch.bfloat16)

        ipex.core.enable_mix_bf16_fp32()
        x_fp32.relu_()
        self.assertEqual(x_fp32.dtype, torch.bfloat16)

        res = (x_fp32 - x_bf16) / x_fp32
        res[torch.isnan(res)] = 0
        zero_base = torch.zeros_like(res)
        self.assertEqual(res, zero_base, 1e-2)

if __name__ == '__main__':
    test = unittest.main()
