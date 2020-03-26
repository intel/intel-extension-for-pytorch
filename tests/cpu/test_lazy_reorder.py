"""Tests for lazy reorder."""
from __future__ import division
from __future__ import print_function

import math
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

device = torch.device("dpcpp:0")
class TestConv(TestCase):
    def test_Conv2d_with_cpu(self):
        torch.manual_seed(1)
        conv_cpu = torch.nn.Conv2d(1, 1, (3, 3))
        conv_dpcpp = torch.nn.Conv2d(1, 1, (3, 3)).to(device=device)

        conv_dpcpp.weight.data = conv_cpu.weight.data.to(device=device)
        conv_dpcpp.bias.data = conv_cpu.bias.data.to(device=device)

        input_cpu = torch.rand((1, 1, 7, 7))
        input_dpcpp = input_cpu.to(device=device)

        out_dpcpp = conv_dpcpp(input_dpcpp)
        out_dpcpp_cpu = out_dpcpp.to('cpu')
        out_cpu = conv_cpu(input_cpu)
        self.assertEqual(out_dpcpp.size(), out_cpu.size())
        self.assertEqual(out_cpu, out_dpcpp_cpu)

    def _seq_conf(self, device):
        torch.manual_seed(1)
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
        res_cpu = self._seq_conf(device='cpu')
        res_dpcpp = self._seq_conf(device=device)
        self.assertEqual(res_cpu, res_dpcpp.to('cpu'))


if __name__ == '__main__':
    test = unittest.main()