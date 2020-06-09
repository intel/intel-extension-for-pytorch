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

import intel_pytorch_extension
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

device = 'dpcpp:0'
#device = 'cpu:0'
SIZE = 100

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

class Conv2dRelu_Fixed(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dRelu_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class CascadedConv2dBnSumRelu(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, **kwargs):
        super(CascadedConv2dBnSumRelu, self).__init__()
        torch.manual_seed(2018)
        self.conv = nn.Conv2d(in_channels, mid_channels, bias=False, **kwargs)
        self.conv1 = nn.Conv2d(
            mid_channels, out_channels, bias=False, padding=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(mid_channels, eps=0.001)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001)

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

    def _test_output(self, model, x):
        modelName = model.__class__.__name__
        core.disable_jit()

        model = model.to('dpcpp').eval()
        x = x.to('dpcpp')
        with torch.no_grad():
            result = model(x)

        script_model = torch.jit.script(model)
        script_model.eval()
        with torch.no_grad():
            sresult = script_model(x)

        self.assertEqual(result, sresult)

        core.enable_jit()
        fused_model = torch.jit.script(model)
        # bn folding, removing it after solve some issue
        core.disable_auto_dnnl()
        fused_model = wrap_cpp_module(torch._C._jit_pass_fold_convbn(fused_model._c))
        core.enable_auto_dnnl()
        # prepack convolution weight
        fused_model = wrap_cpp_module(core._jit_prepack_conv_weight(fused_model._c))
        with torch.no_grad():
            # conv relu fusion, conv sum fusion or conv sum relu fusion
            print(fused_model.graph_for(x))
            fresult = fused_model(x)

        # print(result)
        # print(sresult)
        # print(fresult)

        self.assertEqual(result, fresult)

    def test_output_conv_relu(self):
        self._test_output(
            Conv2dRelu_Fixed(3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 224, 224))

    def test_output_cascaded_conv2d_bn_sum_relu(self):
        self._test_output(
            CascadedConv2dBnSumRelu(3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 224, 224))

    def test_output_linear_relu(self):
        self._test_output(
            LinearRelu(3, 32, bias=True),
            torch.rand(32, 3))
        self._test_output(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3))

if __name__ == '__main__':
    core.enable_auto_dnnl()
    test = unittest.main()
