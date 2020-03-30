"""Tests for lazy reorder."""
from __future__ import division
from __future__ import print_function

import os
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

        ipex.enable_auto_dnnl()
        out_dpcpp = conv_dpcpp(input_dpcpp)

        ipex.disable_auto_dnnl()
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
        ipex.disable_auto_dnnl()
        res_cpu = self._seq_conf(device='cpu')

        ipex.enable_auto_dnnl()
        res_dpcpp = self._seq_conf(device=device)
        self.assertEqual(res_cpu, res_dpcpp.to('cpu'))

class TestBinaryOp(TestCase):
    def _test_dil_add(self, device):
        torch.manual_seed(10)
        a = torch.rand((8, 8)).to(device=device)
        a1 = a[0:2, :]
        a2 = a[4:6, :]
        self.assertEqual(a1.is_contiguous(), True)
        self.assertEqual(a2.is_contiguous(), True)
        a1 += a2
        return a1

    def test_dil_add_(self):
        ipex.enable_auto_dnnl()
        res_dcpp_dnnl = self._test_dil_add("dpcpp:0")

        ipex.disable_auto_dnnl()
        res_dcpp_cpu = self._test_dil_add("dpcpp:0")

        res_cpu = self._test_dil_add("cpu")
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))

    def test_dil_add_scalar(self):
        ipex.enable_auto_dnnl()
        a = torch.rand((8, 8)).to(device=device)
        a += 2

class TestMixOp(TestCase):
    def _test_conv_add_(self, device):
        torch.manual_seed(1)
        conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=device)
        conv_op_input = torch.rand((1, 1, 10, 10)).to(device=device)
        conv_op_putput = conv_op(conv_op_input)
        print(conv_op_putput.size())
        add_src = torch.rand((1, 1, 4, 4)).to(device=device)
        conv_op_putput += add_src
        return conv_op_putput

    def test_conv_add_(self):
        ipex.enable_auto_dnnl()
        res_dcpp_dnnl = self._test_conv_add_("dpcpp:0")

        ipex.disable_auto_dnnl()
        res_dcpp_cpu = self._test_conv_add_("dpcpp:0")

        res_cpu = self._test_conv_add_("cpu")
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))


if __name__ == '__main__':
    test = unittest.main()