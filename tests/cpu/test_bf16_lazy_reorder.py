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

def convert_to_bf16(t):
    last_dim = t.dim() - 1
    out_features = t.size(last_dim)
    in_features = t.size(last_dim)

    t = t.clone().to(device)
    with AutoDNNL(True), AutoMixPrecision(True):
        return F.linear(t, torch.eye(out_features, in_features).to(device))

def _gen_tensor(seed, shape, is_forward=True):
    torch.manual_seed(seed)
    x = torch.randn(shape) * 10
    if is_forward:
        x_cpu = x
        x_auto_mix_inference = x_cpu.to(device=device)
        x_auto_mix_train = copy.deepcopy(x_auto_mix_inference)
        x_man_bf16 = copy.deepcopy(x_auto_mix_inference).to(torch.bfloat16)
        x_auto_mix_train_bf16 = convert_to_bf16(x_cpu)
    else:
        x_cpu = x.clone().requires_grad_()
        x_auto_mix_inference = None
        x_auto_mix_train = x.clone().to(device=device).requires_grad_()
        x_man_bf16 = x.clone().to(device=device).to(torch.bfloat16).requires_grad_()
        x_auto_mix_train_bf16 = convert_to_bf16(x).requires_grad_()

    return x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16

def _gen_op(seed, op, is_bn=False, is_forward=True):
    torch.manual_seed(seed)
    op_cpu = op
    if is_forward:
        op_auto_mix_inference = copy.deepcopy(op_cpu).to(device=device)
    else:
        op_auto_mix_inference = None
    op_auto_mix_train = copy.deepcopy(op_cpu).to(device=device)
    
    # Mean / Variance / ScaleShift of BN cannot be bf16
    if is_bn:
        op_man_bf16 =copy.deepcopy(op_cpu).to(device=device)
    else:
        op_man_bf16 =copy.deepcopy(op_cpu).to(device=device).to(torch.bfloat16)
    op_auto_mix_train_bf16 = copy.deepcopy(op_cpu).to(device=device)
    
    return op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16

class CascadedConvBnSumRelu(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(CascadedConvBnSumRelu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, mid_channels, bias=False, **kwargs)
        self.conv1 = torch.nn.Conv2d(
            mid_channels, out_channels, bias=False, padding=1, **kwargs)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(mid_channels, eps=0.001)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.bn2 = torch.nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        a = self.conv(x)
        a = self.bn(a)
        a = F.relu(a, inplace=True)
        a = self.conv1(a)
        a = self.bn1(a)
        b = self.conv2(x)
        b = self.bn2(b)
        return F.relu(a.add_(b), inplace=True)

def apply(m, fn, args):
    for sub_module in m.children():
        apply(sub_module, fn, args)
    fn(m, args)

class TestTo(TestCase):
    def test_to(self):
        rand_seed = int(get_rand_seed())
        torch.manual_seed(rand_seed)

        m = CascadedConvBnSumRelu(3, 64, 32, kernel_size=3, stride=1)
        m_cpu = copy.deepcopy(m).to("cpu")
        m_data_type = copy.deepcopy(m).to(torch.bfloat16)
        m_auto_mix = copy.deepcopy(m).to(device)
        m_auto_mix_data_type = copy.deepcopy(m).to(device=device, dtype=torch.bfloat16)

        def check_param(t, is_param):
            for param in t.parameters():
                if is_param:
                    self.assertTrue(ipex.core.is_parameter_tensor(param.data))
                else:
                    self.assertFalse(ipex.core.is_parameter_tensor(param.data))

        apply(m_cpu, check_param, False)
        apply(m_data_type, check_param, False)
        apply(m_auto_mix, check_param, True)
        apply(m_auto_mix_data_type, check_param, True)

class TestConv(TestCase):
    def test_Conv2d_with_cpu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        _conv = torch.nn.Conv2d(1, 1, (3, 3))

        conv_man_bf16 =copy.deepcopy(_conv).to(device=device).to(torch.bfloat16)
        conv_auto_mix =copy.deepcopy(_conv).to(device=device)

        _in_cpu = torch.rand((1, 1, 7, 7))
        in_auto_mix = _in_cpu.to(device=device)
        in_man_bf16 = in_auto_mix.to(torch.bfloat16)

        res_cpu_fp32 = _conv(_in_cpu)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = conv_man_bf16(in_man_bf16)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            self.assertEqual(res_cpu_fp32.bfloat16().float(), res_man_bf16, 1e-2)

            with AutoMixPrecision(True):
                self.assertEqual(in_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(in_auto_mix))
                res_auto_bf16 = conv_auto_mix(in_auto_mix)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_bf16))
                self.assertEqual(res_man_bf16.float(), res_auto_bf16.float())

    def test_Conv2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        
        input = torch.rand((1, 1, 7, 7))
        for bias in [True, False]:
            _conv = torch.nn.Conv2d(1, 1, (3, 3), bias=bias)
            conv_man_bf16 =copy.deepcopy(_conv).to(device=device).to(torch.bfloat16)
            conv_auto_mix =copy.deepcopy(_conv).to(device=device)
            _in_cpu = input.clone().requires_grad_()
            in_auto_mix = input.clone().to(device=device).requires_grad_()
            in_man_bf16 = input.clone().to(device=device).to(torch.bfloat16).requires_grad_()
            out_cpu = _conv(_in_cpu).sum()
            out_cpu.backward()
            with AutoDNNL(True), AutoMixPrecision(False, train=True):
                out_man_bf16 = conv_man_bf16(in_man_bf16).sum()
                out_man_bf16.backward()
                self.assertEqual(in_man_bf16.grad.dtype, torch.bfloat16)
                self.assertEqual(_in_cpu.grad.bfloat16().float(), in_man_bf16.grad, 1e-2)

                with AutoMixPrecision(True, train=True):
                    self.assertEqual(in_auto_mix.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(in_auto_mix))
                    out_auto_bf16 = conv_auto_mix(in_auto_mix).sum()
                    out_auto_bf16.backward()
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(in_auto_mix.grad))
                    self.assertEqual(in_man_bf16.grad.float(), in_auto_mix.grad.float())

class TestDeconv(TestCase):
    def test_Deconv2d_with_cpu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)

        _deconv = torch.nn.ConvTranspose2d(2, 3, (3, 3))

        deconv_man_bf16 =copy.deepcopy(_deconv).to(device=device).to(torch.bfloat16)
        deconv_auto_mix =copy.deepcopy(_deconv).to(device=device)

        _in_cpu = torch.rand((1, 2, 7, 7))
        in_auto_mix = _in_cpu.to(device=device)
        in_man_bf16 = in_auto_mix.to(torch.bfloat16)

        res_cpu_fp32 = _deconv(_in_cpu)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = deconv_man_bf16(in_man_bf16)
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            self.assertEqual(res_cpu_fp32.bfloat16().float(), res_man_bf16, 1e-2)

            with AutoMixPrecision(True):
                self.assertEqual(in_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(in_auto_mix))
                res_auto_bf16 = deconv_auto_mix(in_auto_mix)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_bf16))
                self.assertEqual(res_man_bf16.float(), res_auto_bf16.float(), 1e-2)

class TestBatchNorm(TestCase):
    def test_batch_norm2d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(
            rand_seed, (64, 3, 35, 45))

        op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(
            rand_seed, torch.nn.BatchNorm2d(3), is_bn=True)

        ref_cpu = op_cpu(x_cpu)
        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = op_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(res_bf16.float(), res_auto_mix_inference)
            
            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                self.assertEqual(res_auto_mix_train.dtype, torch.float)
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(ref_cpu, res_auto_mix_train)
            
            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16)

    def test_batch_norm2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (64, 3, 35, 45), is_forward=False)
        
        op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.BatchNorm2d(3), is_bn=True, is_forward=False)

        out_cpu = op_cpu(x_cpu).sum()
        out_cpu.backward()
        with AutoDNNL(True), AutoMixPrecision(False, train=True):
            out_man_bf16 = op_man_bf16(x_man_bf16).sum()
            out_man_bf16.backward()
            self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
            self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

            # BW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                out_auto_mix = op_auto_mix(x_auto_mix).sum()
                out_auto_mix.backward()
                self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                self.assertEqual(x_cpu.grad, x_auto_mix.grad)
            
             # BW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                out_auto_mix_bf16.backward()
                self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

    def test_batch_norm3d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(
            rand_seed, (4, 3, 30, 30, 30))

        op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(
            rand_seed, torch.nn.BatchNorm3d(3), is_bn=True)
        
        ref_cpu = op_cpu(x_cpu)
        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = op_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(res_bf16.float(), res_auto_mix_inference)

            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                self.assertEqual(res_auto_mix_train.dtype, torch.float)
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(ref_cpu, res_auto_mix_train)

            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_batch_norm3d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        
        x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (4, 3, 30, 30, 30), is_forward=False)
        
        op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.BatchNorm3d(3), is_bn=True, is_forward=False)

        out_cpu = op_cpu(x_cpu).sum()
        out_cpu.backward()
        with AutoDNNL(True), AutoMixPrecision(False, train=True):
            out_man_bf16 = op_man_bf16(x_man_bf16).sum()
            out_man_bf16.backward()
            self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
            self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

            # BW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                out_auto_mix = op_auto_mix(x_auto_mix).sum()
                out_auto_mix.backward()
                self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                self.assertEqual(x_cpu.grad, x_auto_mix.grad)
            
             # BW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                out_auto_mix_bf16.backward()
                self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

class TestRelu(TestCase):
    def test_relu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (4, 5))

        op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(rand_seed, torch.nn.ReLU(), is_bn=False)

        ref_cpu = op_cpu(x_cpu)
        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = op_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(res_bf16.float(), res_auto_mix_inference)

            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                self.assertEqual(res_auto_mix_train.dtype, torch.float)
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(ref_cpu, res_auto_mix_train)

            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_relu_(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (4, 5))

        x_cpu.relu_()
        with AutoDNNL(True), AutoMixPrecision(False):
            x_man_bf16.relu_()
            self.assertEqual(x_man_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                x_auto_mix_inference.relu_()
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(x_man_bf16.float(), x_auto_mix_inference)

            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                x_auto_mix_train.relu_()
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(x_cpu, x_auto_mix_train)

            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                x_auto_mix_train_bf16.relu_()
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(x_man_bf16.float(), x_auto_mix_train_bf16, 1e-3)

    def test_relu_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (4, 5), is_forward=False)
    
        op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.ReLU(), is_bn=False, is_forward=False)

        out_cpu = op_cpu(x_cpu).sum()
        out_cpu.backward()
        with AutoDNNL(True), AutoMixPrecision(False, train=True):
            out_man_bf16 = op_man_bf16(x_man_bf16).sum()
            out_man_bf16.backward()
            self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
            self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

            # BW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                out_auto_mix = op_auto_mix(x_auto_mix).sum()
                out_auto_mix.backward()
                self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                self.assertEqual(x_cpu.grad, x_auto_mix.grad)
            
             # BW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                out_auto_mix_bf16.backward()
                self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

class TestGelu(TestCase):
    def test_gelu(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (4, 5))

        op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(rand_seed, torch.nn.GELU(), is_bn=False)

        ref_cpu = op_cpu(x_cpu)
        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = op_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(res_bf16.float(), res_auto_mix_inference)

            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                self.assertEqual(res_auto_mix_train.dtype, torch.float)
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(ref_cpu, res_auto_mix_train, 1e-3)

            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_gelu_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (4, 5), is_forward=False)
        
        op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.GELU(), is_bn=False, is_forward=False)

        out_cpu = op_cpu(x_cpu).sum()
        out_cpu.backward()
        with AutoDNNL(True), AutoMixPrecision(False, train=True):
            out_man_bf16 = op_man_bf16(x_man_bf16).sum()
            out_man_bf16.backward()
            self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
            self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

            # BW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                out_auto_mix = op_auto_mix(x_auto_mix).sum()
                out_auto_mix.backward()
                self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                self.assertEqual(x_cpu.grad, x_auto_mix.grad, 1e-3)
            
             # BW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                out_auto_mix_bf16.backward()
                self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

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

    def _gen_binary_tensors(self, rand_seed):
        torch.manual_seed(rand_seed)

        shape = self._gen_shapes()

        x_cpu_a = torch.rand(shape)
        x_cpu_b = torch.rand(shape)

        x_auto_mix_a_infer = x_cpu_a.to(device=device)
        x_auto_mix_b_infer = x_cpu_b.to(device=device)

        x_auto_mix_a = x_cpu_a.to(device=device)
        x_auto_mix_b = x_cpu_b.to(device=device)
        x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
        x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)

        x_auto_mix_bf16_a = convert_to_bf16(x_cpu_a)
        x_auto_mix_bf16_b = convert_to_bf16(x_cpu_b)

        return x_cpu_a, x_cpu_b, x_auto_mix_a_infer, x_auto_mix_b_infer, x_auto_mix_a, x_auto_mix_b, x_man_bf16_a, x_man_bf16_b, x_auto_mix_bf16_a, x_auto_mix_bf16_b

    def test_add(self):
        rand_seed = int(get_rand_seed())

        # generate a 4D tensor by using this seed
        rand_seed = 1591058395950926848
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu_a, x_cpu_b, x_auto_mix_a_infer, x_auto_mix_b_infer, x_auto_mix_a, x_auto_mix_b, x_man_bf16_a, x_man_bf16_b, x_auto_mix_bf16_a, x_auto_mix_bf16_b = self._gen_binary_tensors(rand_seed)
        res_cpu = x_cpu_a + x_cpu_b
        with AutoDNNL(True), AutoMixPrecision(False):
            self.assertEqual(x_man_bf16_a.dtype, torch.bfloat16)
            self.assertEqual(x_man_bf16_b.dtype, torch.bfloat16)
            res_man_bf16 = x_man_bf16_a + x_man_bf16_b
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

            # For inference: reorder the two inputs to bf16 in auto mix precision mode
            with AutoMixPrecision(True, train=False):
                # fp32 + fp32
                self.assertEqual(x_auto_mix_a_infer.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a_infer))
                self.assertEqual(x_auto_mix_b_infer.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b_infer))
                res_auto_mix_infer = x_auto_mix_a_infer + x_auto_mix_b_infer
                self.assertEqual(res_auto_mix_infer.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_infer))

                self.assertEqual(x_auto_mix_a_infer.dtype, torch.float) 
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_a_infer))

                self.assertEqual(x_auto_mix_b_infer.dtype, torch.float) 
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_b_infer))
                
                self.assertEqual(res_auto_mix_infer, res_man_bf16)
            
            # For training: only reorder the second tensor to the data type of the first tensor.
            with AutoMixPrecision(True, train=True):
                # bf16 + bf16
                self.assertEqual(x_auto_mix_bf16_a.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a))
                self.assertEqual(x_auto_mix_bf16_b.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_b))
                res_auto_mix_bf16 = x_auto_mix_bf16_a + x_auto_mix_bf16_b
                self.assertEqual(res_auto_mix_bf16.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_bf16))
                
                self.assertEqual(res_auto_mix_bf16, res_man_bf16)

                # bf16 + fp32
                self.assertEqual(x_auto_mix_bf16_a.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a))
                self.assertEqual(x_auto_mix_b.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b))
                res_auto_mix_reorder_b = x_auto_mix_bf16_a + x_auto_mix_b
                self.assertEqual(res_auto_mix_reorder_b.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_reorder_b))
                
                self.assertEqual(res_auto_mix_reorder_b, res_man_bf16)

                # fp32 + bf16
                self.assertEqual(x_auto_mix_a.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                self.assertEqual(x_auto_mix_bf16_b.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_b))
                res_auto_mix_reorder_a = x_auto_mix_a + x_auto_mix_bf16_b
                self.assertEqual(res_auto_mix_reorder_a.dtype, torch.float)           
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_reorder_a))
                
                self.assertEqual(res_auto_mix_reorder_a, res_cpu, 2e-3)

    def test_mul(self):
        rand_seed = int(get_rand_seed())

        # generate a 4D tensor by using this seed
        rand_seed = 1591058395950926848
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu_a, x_cpu_b, x_auto_mix_a_infer, x_auto_mix_b_infer, x_auto_mix_a, x_auto_mix_b, x_man_bf16_a, x_man_bf16_b, x_auto_mix_bf16_a, x_auto_mix_bf16_b = self._gen_binary_tensors(rand_seed)
        res_cpu = x_cpu_a * x_cpu_b
        with AutoDNNL(True), AutoMixPrecision(False):
            self.assertEqual(x_man_bf16_a.dtype, torch.bfloat16)
            self.assertEqual(x_man_bf16_b.dtype, torch.bfloat16)
            res_man_bf16 = x_man_bf16_a * x_man_bf16_b
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)

            # For inference: reorder the two inputs to bf16 in auto mix precision mode
            with AutoMixPrecision(True, train=False):
                # fp32 * fp32
                self.assertEqual(x_auto_mix_a_infer.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a_infer))
                self.assertEqual(x_auto_mix_b_infer.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b_infer))
                res_auto_mix_infer = x_auto_mix_a_infer * x_auto_mix_b_infer
                self.assertEqual(res_auto_mix_infer.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_infer))

                self.assertEqual(x_auto_mix_a_infer.dtype, torch.float) 
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_a_infer))

                self.assertEqual(x_auto_mix_b_infer.dtype, torch.float) 
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_b_infer))
                
                self.assertEqual(res_auto_mix_infer, res_man_bf16)
            
            # For training: only reorder the second tensor to the data type of the first tensor.
            with AutoMixPrecision(True, train=True):
                # bf16 * bf16
                self.assertEqual(x_auto_mix_bf16_a.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a))
                self.assertEqual(x_auto_mix_bf16_b.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_b))
                res_auto_mix_bf16 = x_auto_mix_bf16_a * x_auto_mix_bf16_b
                self.assertEqual(res_auto_mix_bf16.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_bf16))
                
                self.assertEqual(res_auto_mix_bf16, res_man_bf16)

                # bf16 * fp32
                self.assertEqual(x_auto_mix_bf16_a.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a))
                self.assertEqual(x_auto_mix_b.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b))
                res_auto_mix_reorder_b = x_auto_mix_bf16_a * x_auto_mix_b
                self.assertEqual(res_auto_mix_reorder_b.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_reorder_b))
                
                self.assertEqual(res_auto_mix_reorder_b, res_man_bf16)

                # fp32 * bf16
                self.assertEqual(x_auto_mix_a.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                self.assertEqual(x_auto_mix_bf16_b.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_b))
                res_auto_mix_reorder_a = x_auto_mix_a * x_auto_mix_bf16_b
                self.assertEqual(res_auto_mix_reorder_a.dtype, torch.float)           
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_reorder_a))
                
                self.assertEqual(res_auto_mix_reorder_a, res_cpu, 2e-3)

    def test_mul_(self):
        rand_seed = int(get_rand_seed())
        
        # generate a 4D tensor by using this seed
        rand_seed = 1591058395950926848
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        torch.manual_seed(rand_seed)

        shape = self._gen_shapes()

        x_cpu_a = torch.rand(shape)
        x_cpu_b = torch.rand(shape)

        x_auto_mix_a_infer = x_cpu_a.to(device=device)
        x_auto_mix_b_infer = x_cpu_b.to(device=device)

        x_auto_mix_a = x_cpu_a.to(device=device)
        x_auto_mix_b = x_cpu_b.to(device=device)
        x_man_bf16_a = x_auto_mix_a.to(torch.bfloat16)
        x_man_bf16_b = x_auto_mix_b.to(torch.bfloat16)

        x_auto_mix_bf16_a = convert_to_bf16(x_cpu_a)
        x_auto_mix_bf16_a_ = convert_to_bf16(x_cpu_a)
        x_auto_mix_bf16_b = convert_to_bf16(x_cpu_b)
        
        x_cpu_a *= x_cpu_b
        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = x_man_bf16_a * x_man_bf16_b
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            
            self.assertEqual(x_man_bf16_a.dtype, torch.bfloat16)
            self.assertEqual(x_man_bf16_b.dtype, torch.bfloat16)
            x_man_bf16_a *= x_man_bf16_b
            self.assertEqual(x_man_bf16_a.dtype, torch.bfloat16)
            self.assertEqual(x_man_bf16_a.float(), res_man_bf16.float())

            # For inference: reorder the two inputs to bf16 in auto mix precision mode
            with AutoMixPrecision(True, train=False):
                # fp32 + fp32
                self.assertEqual(x_auto_mix_a_infer.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a_infer))
                self.assertEqual(x_auto_mix_b_infer.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b_infer))
                x_auto_mix_a_infer *= x_auto_mix_b_infer

                self.assertEqual(x_auto_mix_a_infer.dtype, torch.float) 
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_a_infer))

                self.assertEqual(x_auto_mix_b_infer.dtype, torch.float) 
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_b_infer))
                
                self.assertEqual(x_auto_mix_a_infer, res_man_bf16)           

            # For training, only reorder the second tensor to the data type of the first tensor.
            with AutoMixPrecision(True, train=True):
                # bf16 * bf16
                self.assertEqual(x_auto_mix_bf16_a.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a))
                self.assertEqual(x_auto_mix_bf16_b.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_b))
                x_auto_mix_bf16_a *= x_auto_mix_bf16_b
                self.assertEqual(x_auto_mix_bf16_a.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a))
                
                self.assertEqual(x_auto_mix_bf16_a, x_man_bf16_a)

                # bf16 * fp32
                self.assertEqual(x_auto_mix_bf16_a_.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a_))
                self.assertEqual(x_auto_mix_b.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_b))
                x_auto_mix_bf16_a_ *= x_auto_mix_b
                self.assertEqual(x_auto_mix_bf16_a_.dtype, torch.float)           
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_a_))
                
                self.assertEqual(x_auto_mix_bf16_a_, res_man_bf16)

                # fp32 * bf16
                self.assertEqual(x_auto_mix_a.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                self.assertEqual(x_auto_mix_bf16_b.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16_b))
                x_auto_mix_a *= x_auto_mix_bf16_b
                self.assertEqual(x_auto_mix_a.dtype, torch.float)           
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_a))
                
                self.assertEqual(x_auto_mix_a, x_cpu_a, 2e-3)

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

    def test_linear_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        input = torch.randn(3, in_features) * 10
        for bias in [True, False]:
            _linear = torch.nn.Linear(in_features, out_features, bias=bias)
            linear_man_bf16 =copy.deepcopy(_linear).to(device=device).to(torch.bfloat16)
            linear_auto_mix =copy.deepcopy(_linear).to(device=device)
            _in_cpu = input.clone().requires_grad_()
            in_auto_mix = input.clone().to(device=device).requires_grad_()
            in_man_bf16 = input.clone().to(device=device).to(torch.bfloat16).requires_grad_()
            out_cpu = _linear(_in_cpu).sum()
            out_cpu.backward()
            with AutoDNNL(True), AutoMixPrecision(False, train=True):
                out_man_bf16 = linear_man_bf16(in_man_bf16).sum()
                out_man_bf16.backward()
                self.assertEqual(in_man_bf16.grad.dtype, torch.bfloat16)
                self.assertEqual(_in_cpu.grad.bfloat16().float(), in_man_bf16.grad, 2e-2)

                with AutoMixPrecision(True, train=True):
                    self.assertEqual(in_auto_mix.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(in_auto_mix))
                    out_auto_bf16 = linear_auto_mix(in_auto_mix).sum()
                    out_auto_bf16.backward()
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(in_auto_mix.grad))
                    self.assertEqual(in_man_bf16.grad.float(), in_auto_mix.grad.float())

class TestPool(TestCase):
    def test_avg_pool2d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for count_include_pad in [True, False]:
            x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(
                rand_seed, 
                (N, C, 64, 64))
            op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(
                rand_seed, 
                torch.nn.AvgPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=count_include_pad),
                is_bn=False)

            ref_cpu = op_cpu(x_cpu)
            with AutoDNNL(True), AutoMixPrecision(False):
                res_bf16 = op_man_bf16(x_man_bf16)
                self.assertEqual(res_bf16.dtype, torch.bfloat16)

                # FW inference
                with AutoMixPrecision(True, train=False):
                    self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                    res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                    self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                    self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                    self.assertEqual(res_bf16.float(), res_auto_mix_inference)
                
                # FW train (input is not bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_train.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                    res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                    self.assertEqual(res_auto_mix_train.dtype, torch.float)
                    self.assertEqual(x_auto_mix_train.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                    self.assertEqual(ref_cpu, res_auto_mix_train)
                
                # FW train (input is bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                    res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                    self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                    self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                    self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_avg_pool2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        
        for count_include_pad in [True, False]: 
            x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(
                rand_seed, (N, C, 64, 64), 
                is_forward=False)
            
            op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(
                rand_seed,             
                torch.nn.AvgPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=count_include_pad), 
                    is_bn=False, 
                is_forward=False)
            out_cpu = op_cpu(x_cpu).sum()
            out_cpu.backward()
            with AutoDNNL(True), AutoMixPrecision(False, train=True):
                out_man_bf16 = op_man_bf16(x_man_bf16).sum()
                out_man_bf16.backward()
                self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
                self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

                # BW train (input is not bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                    out_auto_mix = op_auto_mix(x_auto_mix).sum()
                    out_auto_mix.backward()
                    self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                    self.assertEqual(x_cpu.grad, x_auto_mix.grad)
                
                # BW train (input is bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                    out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                    out_auto_mix_bf16.backward()        
                    self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                    self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

    def test_avg_pool3d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for count_include_pad in [True, False]:
            x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(
                rand_seed, 
                (N, C, 64, 64, 64))
            op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(
                rand_seed, 
                torch.nn.AvgPool3d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=count_include_pad),
                is_bn=False)

            ref_cpu = op_cpu(x_cpu)
            with AutoDNNL(True), AutoMixPrecision(False):
                res_bf16 = op_man_bf16(x_man_bf16)
                self.assertEqual(res_bf16.dtype, torch.bfloat16)

                # FW inference
                with AutoMixPrecision(True, train=False):
                    self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                    res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                    self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                    self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                    self.assertEqual(res_bf16.float(), res_auto_mix_inference)
                
                # FW train (input is not bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_train.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                    res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                    self.assertEqual(res_auto_mix_train.dtype, torch.float)
                    self.assertEqual(x_auto_mix_train.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                    self.assertEqual(ref_cpu, res_auto_mix_train)
                
                # FW train (input is bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                    res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                    self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                    self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                    self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_avg_pool3d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        
        for count_include_pad in [True, False]:
            
            x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(
                rand_seed, (N, C, 64, 64, 64), 
                is_forward=False)
            
            op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(
                rand_seed,             
                torch.nn.AvgPool3d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    count_include_pad=count_include_pad), 
                    is_bn=False,
                is_forward=False)
            out_cpu = op_cpu(x_cpu).sum()
            out_cpu.backward()
            with AutoDNNL(True), AutoMixPrecision(False, train=True):
                out_man_bf16 = op_man_bf16(x_man_bf16).sum()
                out_man_bf16.backward()
                self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
                self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

                # BW train (input is not bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                    out_auto_mix = op_auto_mix(x_auto_mix).sum()
                    out_auto_mix.backward()
                    self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                    self.assertEqual(x_cpu.grad, x_auto_mix.grad)
                
                # BW train (input is bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                    out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                    out_auto_mix_bf16.backward()        


                    self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                    self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

    def test_adaptive_avg_pool2d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (N, C, 224, 224))
        op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(rand_seed, torch.nn.AdaptiveAvgPool2d(7), is_bn=False)

        ref_cpu = op_cpu(x_cpu)
        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = op_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(res_bf16.float(), res_auto_mix_inference)
            
            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                self.assertEqual(res_auto_mix_train.dtype, torch.float)
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(ref_cpu, res_auto_mix_train)
    
            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_adaptive_avg_pool2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
            
        x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (N, C, 224, 224), is_forward=False)
        
        op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.AdaptiveAvgPool2d(7), is_bn=False, is_forward=False)
        out_cpu = op_cpu(x_cpu).sum()
        out_cpu.backward()
        with AutoDNNL(True), AutoMixPrecision(False, train=True):
            out_man_bf16 = op_man_bf16(x_man_bf16).sum()
            out_man_bf16.backward()
            self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
            self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

            # BW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                out_auto_mix = op_auto_mix(x_auto_mix).sum()
                out_auto_mix.backward()
                self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                self.assertEqual(x_cpu.grad, x_auto_mix.grad)
            
            # BW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                out_auto_mix_bf16.backward()        

                self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

    def test_max_pool2d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                for ceil_mode in [False, True]:
                    x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(
                        rand_seed, 
                        (N, C, H, W))
                    op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(
                        rand_seed, 
                        torch.nn.MaxPool2d(
                            kernel_size=3 if not ceil_mode else 7,
                            stride=stride,
                            padding=1,
                            ceil_mode=ceil_mode),
                        is_bn=False)

                    ref_cpu = op_cpu(x_cpu)
                    with AutoDNNL(True), AutoMixPrecision(False):
                        res_bf16 = op_man_bf16(x_man_bf16)
                        self.assertEqual(res_bf16.dtype, torch.bfloat16)

                        # FW inference
                        with AutoMixPrecision(True, train=False):
                            self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                            res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                            self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                            self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                            self.assertEqual(res_bf16.float(), res_auto_mix_inference)

                        # FW train (input is not bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix_train.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                            res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                            self.assertEqual(res_auto_mix_train.dtype, torch.float)
                            self.assertEqual(x_auto_mix_train.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                            self.assertEqual(ref_cpu, res_auto_mix_train)

                        # FW train (input is bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                            res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                            self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                            self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                            self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_max_pool2d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                for ceil_mode in [False, True]:
                    x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (N, C, H, W), is_forward=False)
                    op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(
                        rand_seed, 
                        torch.nn.MaxPool2d(
                            kernel_size=3 if not ceil_mode else 7,
                            stride=stride,
                            padding=1,
                            ceil_mode=ceil_mode),
                        is_bn=False,
                        is_forward=False)

                    out_cpu = op_cpu(x_cpu).sum()
                    out_cpu.backward()
                    with AutoDNNL(True), AutoMixPrecision(False, train=True):
                        out_man_bf16 = op_man_bf16(x_man_bf16).sum()
                        out_man_bf16.backward()
                        self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
                        
                        # BW train (input is not bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                            out_auto_mix = op_auto_mix(x_auto_mix).sum()
                            out_auto_mix.backward()
                            self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                            self.assertEqual(x_cpu.grad, x_auto_mix.grad)
                        
                        # BW train (input is bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                            out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                            out_auto_mix_bf16.backward()        

                            self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                            self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)
                        
    def test_max_pool3d(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
                for ceil_mode in [False, True]:
                    x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (N, C, D, H, W))
                    op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(
                        rand_seed, 
                        torch.nn.MaxPool3d(
                            kernel_size=3 if not ceil_mode else 7,
                            stride=stride,
                            padding=1,
                            ceil_mode=ceil_mode),
                        is_bn=False)
                    ref_cpu = op_cpu(x_cpu)
                    with AutoDNNL(True), AutoMixPrecision(False):
                        res_bf16 = op_man_bf16(x_man_bf16)
                        self.assertEqual(res_bf16.dtype, torch.bfloat16)

                        # FW inference
                        with AutoMixPrecision(True, train=False):
                            self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                            res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                            self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                            self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                            self.assertEqual(res_bf16.float(), res_auto_mix_inference)

                        # FW train (input is not bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix_train.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                            res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                            self.assertEqual(res_auto_mix_train.dtype, torch.float)
                            self.assertEqual(x_auto_mix_train.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                            self.assertEqual(ref_cpu, res_auto_mix_train)

                        # FW train (input is bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                            res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                            self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                            self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                            self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_max_pool3d_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
                for ceil_mode in [False, True]:
                    x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (N, C, D, H, W), is_forward=False)
                    op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(
                        rand_seed, 
                        torch.nn.MaxPool3d(
                            kernel_size=3 if not ceil_mode else 7,
                            stride=stride,
                            padding=1,
                            ceil_mode=ceil_mode),
                        is_bn=False,
                        is_forward=False)

                    out_cpu = op_cpu(x_cpu).sum()
                    out_cpu.backward()
                    with AutoDNNL(True), AutoMixPrecision(False, train=True):
                        out_man_bf16 = op_man_bf16(x_man_bf16).sum()
                        out_man_bf16.backward()
                        self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)

                        # BW train (input is not bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                            out_auto_mix = op_auto_mix(x_auto_mix).sum()
                            out_auto_mix.backward()
                            self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                            self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                            self.assertEqual(x_cpu.grad, x_auto_mix.grad)
                        
                        # BW train (input is bf16 dil tensor)
                        with AutoMixPrecision(True, train=True):
                            self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                            out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                            out_auto_mix_bf16.backward()        

                            self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                            self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                            self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

class TestIndexSelect(TestCase):
    def test_index_select(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        x_auto_mix = torch.randn(3, 4, 5, dtype=torch.float32, device=device) * 10

        indices = torch.tensor([0, 2]).to(device=device)
        index_select_x_auto_mix = copy.deepcopy(x_auto_mix).to(device=device)
        index_select_x_man_mix = copy.deepcopy(x_auto_mix).to(device=device).to(torch.bfloat16)

        with AutoDNNL(True), AutoMixPrecision(False):
            res_man_bf16 = index_select_x_man_mix + index_select_x_man_mix
            self.assertEqual(res_man_bf16.dtype, torch.bfloat16)
            res_idx_select_man = torch.index_select(res_man_bf16, 0, indices)
            self.assertEqual(res_idx_select_man.dtype, torch.bfloat16)

            with AutoMixPrecision(True):
                res_auto_mix = index_select_x_auto_mix + index_select_x_auto_mix
                self.assertEqual(res_auto_mix.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix))
                res_idx_select_auto = torch.index_select(res_auto_mix, 0, indices)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_idx_select_auto))
                self.assertEqual(res_idx_select_auto, res_idx_select_man.float())

class TestSoftMax(TestCase):
    def test_softmax(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        ndim = 3        
        for dim in range(ndim):
            x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (3, 4, 5))

            op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(rand_seed, torch.nn.Softmax(dim=dim), is_bn=False)

            ref_cpu = op_cpu(x_cpu)
            with AutoDNNL(True), AutoMixPrecision(False):
                res_bf16 = op_man_bf16(x_man_bf16)
                self.assertEqual(res_bf16.dtype, torch.bfloat16)

                # FW inference
                with AutoMixPrecision(True, train=False):
                    self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                    res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                    self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                    self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                    self.assertEqual(res_bf16.float(), res_auto_mix_inference)

                # FW train (input is not bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_train.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                    res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                    self.assertEqual(res_auto_mix_train.dtype, torch.float)
                    self.assertEqual(x_auto_mix_train.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                    self.assertEqual(ref_cpu, res_auto_mix_train)

                # FW train (input is bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                    res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                    self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                    self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                    self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_softmax_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        
        ndim = 3
        for dim in range(ndim):
            x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (3, 4, 5), is_forward=False)
            
            op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.Softmax(dim=dim), is_bn=False, is_forward=False)

            out_cpu = op_cpu(x_cpu).sum()
            out_cpu.backward()
            with AutoDNNL(True), AutoMixPrecision(False, train=True):
                out_man_bf16 = op_man_bf16(x_man_bf16).sum()
                out_man_bf16.backward()
                self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
                self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

                # BW train (input is not bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                    out_auto_mix = op_auto_mix(x_auto_mix).sum()
                    out_auto_mix.backward()
                    self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                    self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                    self.assertEqual(x_cpu.grad, x_auto_mix.grad)
                
                # BW train (input is bf16 dil tensor)
                with AutoMixPrecision(True, train=True):
                    self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                    self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                    out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                    out_auto_mix_bf16.backward()
                    self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                    # TODO
                    # grady and y both fp32 after .sum()
                    # self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                    # self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

class TestSigmoid(TestCase):
    def test_sigmoid(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))

        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (4, 5))

        op_cpu, op_auto_mix_inference, op_auto_mix_train, op_man_bf16, op_auto_mix_train_bf16 = _gen_op(rand_seed, torch.nn.Sigmoid(), is_bn=False)

        ref_cpu = op_cpu(x_cpu)
        with AutoDNNL(True), AutoMixPrecision(False):
            res_bf16 = op_man_bf16(x_man_bf16)
            self.assertEqual(res_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                res_auto_mix_inference = op_auto_mix_inference(x_auto_mix_inference)
                self.assertEqual(res_auto_mix_inference.dtype, torch.float)
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_inference))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(res_bf16.float(), res_auto_mix_inference)
            
            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                res_auto_mix_train = op_auto_mix_train(x_auto_mix_train)
                self.assertEqual(res_auto_mix_train.dtype, torch.float)
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(res_auto_mix_train))
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(ref_cpu, res_auto_mix_train)
            
            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                res_auto_mix_train_bf16 = op_auto_mix_train_bf16(x_auto_mix_train_bf16)
                self.assertEqual(res_auto_mix_train_bf16.dtype, torch.float)
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(res_auto_mix_train_bf16))
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(res_bf16.float(), res_auto_mix_train_bf16, 1e-3)

    def test_sigmoid_(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        x_cpu, x_auto_mix_inference, x_auto_mix_train, x_man_bf16, x_auto_mix_train_bf16 = _gen_tensor(rand_seed, (4, 5))

        x_cpu.sigmoid_()
        with AutoDNNL(True), AutoMixPrecision(False):
            x_man_bf16.sigmoid_()
            self.assertEqual(x_man_bf16.dtype, torch.bfloat16)

            # FW inference
            with AutoMixPrecision(True, train=False):
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                x_auto_mix_inference.sigmoid_()
                self.assertEqual(x_auto_mix_inference.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_inference))
                self.assertEqual(x_man_bf16.float(), x_auto_mix_inference)
            
            # FW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                x_auto_mix_train.sigmoid_()
                self.assertEqual(x_auto_mix_train.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix_train))
                self.assertEqual(x_cpu, x_auto_mix_train)
            
            # FW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                x_auto_mix_train_bf16.sigmoid_()
                self.assertEqual(x_auto_mix_train_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_train_bf16))
                self.assertEqual(x_man_bf16.float(), x_auto_mix_train_bf16, 1e-3)

    def test_sigmoid_backward(self):
        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        
        x_cpu, _, x_auto_mix, x_man_bf16, x_auto_mix_bf16 = _gen_tensor(rand_seed, (4, 5), is_forward=False)
        
        op_cpu, _, op_auto_mix, op_man_bf16, op_auto_mix_bf16 = _gen_op(rand_seed, torch.nn.Sigmoid(), is_bn=False, is_forward=False)
        
        out_cpu = op_cpu(x_cpu).sum()
        out_cpu.backward()
        with AutoDNNL(True), AutoMixPrecision(False, train=True):
            out_man_bf16 = op_man_bf16(x_man_bf16).sum()
            out_man_bf16.backward()
            self.assertEqual(x_man_bf16.grad.dtype, torch.bfloat16)
            self.assertEqual(x_cpu.grad.bfloat16().float(), x_man_bf16.grad, 1e-2)

            # BW train (input is not bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix))
                out_auto_mix = op_auto_mix(x_auto_mix).sum()
                out_auto_mix.backward()
                self.assertEqual(x_auto_mix.grad.dtype, torch.float)
                self.assertFalse(ipex.core.is_bf16_dil_tensor(x_auto_mix.grad))
                self.assertEqual(x_cpu.grad, x_auto_mix.grad)
            
             # BW train (input is bf16 dil tensor)
            with AutoMixPrecision(True, train=True):
                self.assertEqual(x_auto_mix_bf16.dtype, torch.float)
                self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16))
                out_auto_mix_bf16 = op_auto_mix_bf16(x_auto_mix_bf16).sum()
                out_auto_mix_bf16.backward()
                self.assertEqual(x_auto_mix_bf16.grad.dtype, torch.float)
                # TODO
                # grady and y both fp32 after .sum()
                # self.assertTrue(ipex.core.is_bf16_dil_tensor(x_auto_mix_bf16.grad))
                # self.assertEqual(x_man_bf16.grad.float(), x_auto_mix_bf16.grad)

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
