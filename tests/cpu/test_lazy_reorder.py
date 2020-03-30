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
    def _test_conv_add_(self, device, rand_seed):
        torch.manual_seed(rand_seed)
        conv_op = torch.nn.Conv2d(1, 1, (7, 7)).to(device=device)
        conv_op_input = torch.rand((1, 1, 10, 10)).to(device=device)
        conv_op_putput = conv_op(conv_op_input)
        add_src = torch.rand((1, 1, 4, 4)).to(device=device)
        conv_op_putput += add_src
        return conv_op_putput

    def test_conv_add_(self):
        ipex.enable_auto_dnnl()
        rand_seed = int(get_rand_seed())
        print("test_dil_add_ rand sed: {}".format(rand_seed))
        res_dcpp_dnnl = self._test_conv_add_("dpcpp:0", rand_seed)

        ipex.disable_auto_dnnl()
        res_dcpp_cpu = self._test_conv_add_("dpcpp:0", rand_seed)

        res_cpu = self._test_conv_add_("cpu", rand_seed)
        self.assertEqual(res_cpu, res_dcpp_cpu.to('cpu'))
        self.assertEqual(res_cpu, res_dcpp_dnnl.to('cpu'))


if __name__ == '__main__':
    test = unittest.main()
