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
        ipex.enable_mix_bf16_fp32()
        self.assertEqual(input_dpcpp.dtype, torch.float)
        out_dpcpp = conv_dpcpp(input_dpcpp)
        out_cpu = conv_cpu(input_cpu)
        self.assertEqual(out_dpcpp.dtype, torch.bfloat16)
        self.assertEqual(out_dpcpp, out_cpu, 1e-2)

class TestBatchNorm(TestCase):
    def test_batch_norm2d(self):
        ipex.enable_auto_dnnl()
        ipex.enable_mix_bf16_fp32()
        rand_seed = int(get_rand_seed())
        rand_seed = 1
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)

        bn = torch.nn.BatchNorm2d(3)
        ipex.enable_mix_bf16_fp32()
        bn_dpcpp_auto_bf16 =copy.deepcopy(bn).to(device=device)
        res_auto_bf16 = bn_dpcpp_auto_bf16(x_dpcpp)

        ipex.disable_mix_bf16_fp32()
        bn_dpcpp_man_bf16 =copy.deepcopy(bn).to(device=device).to(torch.bfloat16)
        res_man_bf16 = bn_dpcpp_man_bf16(x_dpcpp.to(torch.bfloat16))

        self.assertEqual(res_man_bf16.float(), res_auto_bf16.float())
        self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
        self.assertEqual(res_auto_bf16.dtype, torch.bfloat16)

    def test_batch_norm3d(self):
        ipex.enable_auto_dnnl()
        # ipex.enable_mix_bf16_fp32()
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_cpu = torch.randn(4, 3, 30, 30, 30, dtype=torch.float32) * 10
        x_dpcpp = x_cpu.to(device=device)

        bn = torch.nn.BatchNorm3d(3)
        bn_dpcpp = copy.deepcopy(bn).to(device=device)
        self.assertEqual(bn(x_cpu), bn_dpcpp(x_dpcpp))

if __name__ == '__main__':
    test = unittest.main()
