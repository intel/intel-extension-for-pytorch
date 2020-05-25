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

if __name__ == '__main__':
    test = unittest.main()
