from __future__ import division
from __future__ import print_function

'''
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.
'''

"""Tests for rn50."""

import math
import random
import unittest
from functools import reduce

import torch
import torch.nn as nn
from torch.jit._recursive import wrap_cpp_module
import copy

import intel_pytorch_extension as ipex
from intel_pytorch_extension import core

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

device = ipex.DEVICE
#device = 'cpu:0'
SIZE = 100


conv_module = {2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}
bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}

class ConvBatchNorm_Fixed(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvRelu_Fixed(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvRelu_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class ConvSum(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv1(x)
        return a+b

class CascadedConvBnSumRelu(nn.Module):
    def __init__(self, dim, in_channels, mid_channels, out_channels, **kwargs):
        super(CascadedConvBnSumRelu, self).__init__()
        torch.manual_seed(2018)
        self.conv = conv_module[dim](in_channels, mid_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](
            mid_channels, out_channels, bias=False, padding=1, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](mid_channels, eps=0.001)
        self.bn1 = bn_module[dim](out_channels, eps=0.001)
        self.bn2 = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        a = self.conv(x)
        a = self.bn(a)
        a = F.relu(a, inplace=True)
        a = self.conv1(a)
        a = self.bn1(a)
        b = self.conv2(x)
        b = self.bn2(b)
        return F.relu(a.add_(b), inplace=True)

class LinearRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearRelu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.linear(x), inplace=True)

class Tester(TestCase):

    def _test_output(self, model, x, kind_in_graph=None, kind_not_in_graph=None):
        modelName = model.__class__.__name__
        core.disable_jit_opt()
        core.disable_mix_bf16_fp32()

        model = model.to(device).eval()
        x = x.to(device)
        with torch.no_grad():
            result = model(x)

        script_model = torch.jit.script(model)
        script_model.eval()

        trace_model = torch.jit.trace(model, x)
        trace_model.eval()
        with torch.no_grad():
            sresult = script_model(x)
            tresult = trace_model(x)

        self.assertEqual(result, sresult)
        self.assertEqual(result, tresult)

        core.enable_jit_opt()
        script_fused_model = torch.jit.script(model)
        trace_fused_model = torch.jit.trace(model, x)
        with torch.no_grad():
            # conv relu fusion, conv sum fusion or conv sum relu fusion
            script_graph =  script_fused_model.graph_for(x)
            fused_sresult = script_fused_model(x)

            trace_graph = trace_fused_model.graph_for(x)
            fused_tresult = trace_fused_model(x)

        self.assertEqual(result, fused_sresult)
        self.assertEqual(result, fused_tresult)

        # check if the fused node exists in the graph
        if kind_in_graph is not None:
            self.assertTrue(any(n.kind() == kind_in_graph for n in script_graph.nodes()))
            self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))
        
        # check if certain node does not exist in the graph
        if kind_not_in_graph is not None:
            self.assertTrue(all(n.kind() != kind_not_in_graph for n in script_graph.nodes()))
            self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))


    def _test_output_bf16(self, model, x, kind_in_graph=None, kind_not_in_graph=None, prec=None):
        modelName = model.__class__.__name__

        core.enable_auto_dnnl()
        core.enable_jit_opt()
        core.enable_mix_bf16_fp32()
        
        model = model.to(ipex.DEVICE).eval()
        x = x.to(ipex.DEVICE)
        x2 = x.clone()
        x3 = x.clone()
        
        script_fused_model = torch.jit.script(copy.deepcopy(model))
        trace_fused_model = torch.jit.trace(copy.deepcopy(model), x3)


        with torch.no_grad():
            # bf16, native path
            result = model(x)
            # bf16, jit script path
            script_graph =  script_fused_model.graph_for(x2)
            fused_sresult = script_fused_model(x2)
            # bf 16, jit trace path
            trace_graph = trace_fused_model.graph_for(x3)
            fused_tresult = trace_fused_model(x3)

        self.assertEqual(fused_sresult, result, prec=prec)
        self.assertEqual(fused_tresult, result, prec=prec)

        # check if the fused node exists in the graph
        if kind_in_graph is not None:
            self.assertTrue(any(n.kind() == kind_in_graph for n in script_graph.nodes()))
            self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))
        
        # check if certain node does not exist in the graph
        if kind_not_in_graph is not None:
            self.assertTrue(all(n.kind() != kind_not_in_graph for n in script_graph.nodes()))
            self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))
    

    def test_output_conv_bn_2d(self):
        self._test_output(
            ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph="aten::batch_norm",)
        self._test_output_bf16(
            ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)


    def test_output_conv_bn_3d(self):
        self._test_output(
            ConvBatchNorm_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 112, 112, 112),
            kind_in_graph="aten::conv3d",
            kind_not_in_graph="aten::batch_norm",)
        self._test_output_bf16(
            ConvBatchNorm_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 112, 112, 112),
            kind_in_graph="aten::conv3d",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)


    def test_output_conv_relu_2d(self):
        self._test_output(
            ConvRelu_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224),
            kind_in_graph="ipex::conv2d_relu")
        self._test_output_bf16(
            ConvRelu_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224),
            kind_in_graph="ipex::conv2d_relu")


    def test_output_conv_relu_3d(self):
        self._test_output(
            ConvRelu_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 112, 112, 112),
            kind_in_graph="ipex::conv3d_relu")
        self._test_output_bf16(
            ConvRelu_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 112, 112, 112),
            kind_in_graph="ipex::conv3d_relu")


    def test_output_conv_sum_2d(self):
        self._test_output(
            ConvSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224),
            kind_in_graph="ipex::conv2d_sum")
        self._test_output_bf16(
            ConvSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224),
            kind_in_graph="ipex::conv2d_sum",
            prec=0.04)


    def test_output_conv_sum_3d(self):
        self._test_output(
            ConvSum(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 112, 112, 112),
            kind_in_graph="ipex::conv3d_sum")
        self._test_output_bf16(
            ConvSum(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 112, 112, 112),
            kind_in_graph="ipex::conv3d_sum",
            prec=0.04)


    def test_output_cascaded_conv_bn_sum_relu_2d(self):
        self._test_output(
            CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 224, 224),
            kind_in_graph="ipex::conv2d_sum_relu",
            kind_not_in_graph="aten::batch_norm")
        self._test_output_bf16(
            CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 224, 224),
            kind_in_graph="ipex::conv2d_sum_relu",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)


    def test_output_cascaded_conv_bn_sum_relu_3d(self):
        self._test_output(
            CascadedConvBnSumRelu(3, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 112, 112, 112),
            kind_in_graph="ipex::conv3d_sum_relu",
            kind_not_in_graph="aten::batch_norm",)
        self._test_output_bf16(
            CascadedConvBnSumRelu(3, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 112, 112, 112),
            kind_in_graph="ipex::conv3d_sum_relu",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)


    def test_output_linear_relu(self):
        self._test_output(
            LinearRelu(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_relu")
        self._test_output_bf16(
            LinearRelu(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_relu")
        self._test_output(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_relu")
        self._test_output_bf16(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_relu")


if __name__ == '__main__':
    torch.manual_seed(2020)
    core.enable_auto_dnnl()
    test = unittest.main()
