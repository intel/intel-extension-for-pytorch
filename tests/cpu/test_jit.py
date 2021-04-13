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

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

# device = ipex.DEVICE
device = 'cpu:0'
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

class BatchNormConv_Fixed(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(BatchNormConv_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](in_channels, eps=0.001)

    def forward(self, x):
        return self.conv(self.bn(x))

class BatchNorm_Conv_BatchNorm(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(BatchNorm_Conv_BatchNorm, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn1 = bn_module[dim](in_channels, eps=0.001)
        self.bn2 = bn_module[dim](out_channels, eps=0.001)
    def forward(self, x):
        return self.bn2(self.conv(self.bn1(x)))

class ConvReshapeBatchNorm(nn.Module):
    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(ConvReshapeBatchNorm, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.dest_shape = dest_shape
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](dest_shape[1], eps=0.001)

    def forward(self, x):
        conv_output = self.conv(x)
        return self.bn(torch.reshape(conv_output, self.dest_shape)) 

class Conv_Conv_Concat(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(Conv_Conv_Concat, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
    def forward(self, x):
        return torch.cat((self.conv1(x),self.conv2(x)))

class ConvRelu_Fixed(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvRelu_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class Conv_Relu_Add(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(Conv_Relu_Add, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
    def forward(self, x):
        return torch.add(F.relu(self.conv(x), inplace=True),self.conv(x))

class Conv_Bn_Relu(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(Conv_Bn_Relu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvReshapeRelu(nn.Module):
    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(ConvReshapeRelu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.dest_shape = dest_shape
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return F.relu(torch.reshape(self.conv(x), self.dest_shape), inplace=True)

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

class ConvReshapeSum(nn.Module):
    def __init__(self, dim, in_channels, out_channels, dest_shape, **kwargs):
        super(ConvReshapeSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.dest_shape = dest_shape
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
    def forward(self, x):
        a=torch.reshape(self.conv1(x), self.dest_shape)
        b=torch.reshape(self.conv2(x), self.dest_shape)
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
 
class LinearGelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearGelu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.gelu(self.linear(x))

class LinearAdd(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearAdd, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return torch.add(self.linear(x),self.linear(x))

class Linear_Reshape_Relu(nn.Module):
    def __init__(self, in_channels, out_channels,dest_shape, **kwargs):
        super(Linear_Reshape_Relu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.dest_shape = dest_shape
    def forward(self, x):
        return F.relu(torch.reshape(self.linear(x),self.dest_shape), inplace=True)

class LinearSigmoid(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearSigmoid, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class LinearBn(nn.Module):
    def __init__(self,dim,in_channels, out_channels, **kwargs):
        super(LinearBn, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.bn = bn_module[dim](1, eps=0.001)
    def forward(self, x):
        return self.bn(self.linear(x))

class Linear_Reshape_Bn(nn.Module):
    def __init__(self,dim,in_channels, out_channels,dest_shape,**kwargs):
        super(Linear_Reshape_Bn, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.bn = bn_module[dim](1, eps=0.001)
        self.dest_shape = dest_shape
    def forward(self, x):
        return self.bn(torch.reshape(self.linear(x),self.dest_shape))

class ConvSumInDiffBlock(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvSumInDiffBlock, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
    
    def forward(self, x):
        y = self.conv(x)
        if y.size(1) != x.size(1):
            y += F.pad(x,
                       (0, 0, 0, 0, 0, y.size(1) - x.size(1)), 'constant', 0.)
        else:
            y += x
        return y

class ConvSwishOutplace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishOutplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a1 = self.conv2d(x)
        b1 = torch.sigmoid(a1)
        c1 = torch.mul(a1, b1)

        return c1

class ConvSwishInplace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishInplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a = self.conv2d(x)
        b = torch.sigmoid(a)
        res = a.mul_(b)
        return res

class ConvSigmoidOutplace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSigmoidOutplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a = self.conv2d(x)
        b = torch.sigmoid(a)
        c = torch.add(b, b)
        return c

class ConvSigmoidInplace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSigmoidInplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a = self.conv2d(x)
        b = torch.sigmoid_(a)
        c = torch.add(b, b)
        return c

class ConvHardtanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, inplace=False):
        super(ConvHardtanh, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)
        self.hardtanh = nn.Hardtanh(inplace=inplace)

    def forward(self, x):
        a = self.conv2d(x)
        b = self.hardtanh(a)
        c = torch.add(b, b)
        return c

class ConvElu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, inplace=False):
        super(ConvElu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)
        self.elu = nn.ELU(inplace=inplace)

    def forward(self, x):
        a = self.conv2d(x)
        b = self.elu(a)
        c = torch.add(b, b)
        return c

class ChannelShuffle(nn.Module):
    def __init__(self, batchsize, num_channels, height, width, groups):
        super(ChannelShuffle, self).__init__()
        self.batchsize = batchsize
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.groups = groups

    def forward(self, x):
        channels_per_group = self.num_channels // self.groups
        x = x.view(self.batchsize, self.groups, channels_per_group, self.height, self.width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(self.batchsize, -1, self.height, self.width)
        return x


class Tester(TestCase):

    def _test_output(self, model, x, kind_in_graph=None, kind_not_in_graph=None):
        modelName = model.__class__.__name__
        core.disable_jit_opt()
        # core.disable_mix_bf16_fp32()

        model = model.eval()
        if x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
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
            # print(script_graph)
            fused_sresult = script_fused_model(x)

            trace_graph = trace_fused_model.graph_for(x)
            # print(trace_graph)
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

        # core.enable_auto_dnnl()
        core.enable_jit_opt()
        # core.enable_mix_bf16_fp32()

        model = model.eval()
        if x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
        x2 = x.clone()
        x3 = x.clone()

        with ipex.amp.autocast(enabled=True, configure=ipex.conf.AmpConf(torch.bfloat16)), torch.no_grad():
            # bf16, native path
            result = model(x)
            # script_fused_model = torch.jit.script(copy.deepcopy(model))
            trace_fused_model = torch.jit.trace(copy.deepcopy(model), x3)
            # bf16, jit script path
            # script_graph =  script_fused_model.graph_for(x2)
            # fused_sresult = script_fused_model(x2)
            # bf 16, jit trace path
            trace_graph = trace_fused_model.graph_for(x3)
            fused_tresult = trace_fused_model(x3)

        # disable mix_bf16_fp32 when the calculation is done
        # to avoid affecting other scripts
        # core.disable_mix_bf16_fp32()

        # self.assertEqual(fused_sresult, result, prec=prec)
        self.assertEqual(fused_tresult, result, prec=prec)
        # self.assertEqual(result.dtype, torch.bfloat16)
        # self.assertEqual(fused_sresult.dtype, torch.bfloat16)
        self.assertEqual(fused_tresult.dtype, torch.bfloat16)

        # check if the fused node exists in the graph
        if kind_in_graph is not None:
            # self.assertTrue(any(n.kind() == kind_in_graph for n in script_graph.nodes()))
            self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

        # check if certain node does not exist in the graph
        if kind_not_in_graph is not None:
            # self.assertTrue(all(n.kind() != kind_not_in_graph for n in script_graph.nodes()))
            self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))

    def test_conv2d_fusion(self):
        # ipex.core.disable_jit_opt()
        batch_size = 32
        out_channels = 64
        in_channels = 3
        kernel_size = 3
        image_size = 64
        self._test_output(
            ConvSwishOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_swish")
        self._test_output_bf16(
            ConvSwishOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_swish",
            prec=0.02)
        self._test_output(
            ConvSwishInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_swish")
        self._test_output_bf16(
            ConvSwishInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_swish",
            prec=0.02)
        self._test_output(
            ConvSigmoidOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_sigmoid")
        self._test_output_bf16(
            ConvSigmoidOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_sigmoid",
            prec=0.02)
        self._test_output(
            ConvSigmoidInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_sigmoid")
        self._test_output_bf16(
            ConvSigmoidInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_sigmoid",
            prec=0.02)
        self._test_output(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size, True),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_clamp")
        self._test_output_bf16(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size, True),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_clamp",
            prec=0.02)
        self._test_output(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_clamp")
        self._test_output_bf16(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_clamp",
            prec=0.02)
        self._test_output(
            ConvElu(in_channels, out_channels, kernel_size, image_size, True),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_elu")
        # self._test_output_bf16(
        #     ConvElu(in_channels, out_channels, kernel_size, image_size, True),
        #     torch.randn(batch_size, in_channels, image_size, image_size),
        #     kind_in_graph="ipex::conv2d_elu",
        #     prec=0.02)
        self._test_output(
            ConvElu(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex::conv2d_elu")
        # self._test_output_bf16(
        #     ConvElu(in_channels, out_channels, kernel_size, image_size),
        #     torch.randn(batch_size, in_channels, image_size, image_size),
        #     kind_in_graph="ipex::conv2d_elu",
        #     prec=0.02)

    def test_output_conv_bn_2d(self):
        self._test_output(
            ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph="aten::batch_norm",)
        self._test_output_bf16(
            ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)

    def test_output_bn_conv_2d(self):
        self._test_output(
            BatchNormConv_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::batch_norm",
            kind_not_in_graph=None,)

    def test_output_bn_conv_bn(self):
        self._test_output(
            BatchNorm_Conv_BatchNorm(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::batch_norm",
            kind_not_in_graph=None,)

    def test_output_conv_reshape_bn_2d(self):
        self._test_output(
            ConvReshapeBatchNorm(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::batch_norm",
            kind_not_in_graph=None,)

    def test_output_conv_conv_concate(self):
        self._test_output(
            Conv_Conv_Concat(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph=None,)

    def test_output_conv_relu_add(self):
        self._test_output(
            Conv_Relu_Add(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_relu")

    def test_output_conv_bn_relu(self):
        self._test_output(
            Conv_Bn_Relu(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_relu")

    def test_output_conv_reshape_relu(self):
        self._test_output(
            ConvReshapeRelu(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph="ipex::conv2d_relu",)

    def test_output_conv_reshape_sum(self):
        self._test_output(
            ConvReshapeSum(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="aten::conv2d",
            kind_not_in_graph="ipex::conv2d_sum",)

    def test_output_conv_bn_3d(self):
        self._test_output(
            ConvBatchNorm_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="aten::conv3d",
            kind_not_in_graph="aten::batch_norm")
        self._test_output_bf16(
            ConvBatchNorm_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="aten::conv3d",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)

    def test_output_conv_relu_2d(self):
        self._test_output(
            ConvRelu_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_relu")
        self._test_output_bf16(
            ConvRelu_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_relu",
            prec=0.02)

    def test_output_conv_relu_3d(self):
        self._test_output(
            ConvRelu_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_relu")
        self._test_output_bf16(
            ConvRelu_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_relu",
            prec=0.02)

    def test_output_conv_sum_2d(self):
        self._test_output(
            ConvSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_sum")
        self._test_output_bf16(
            ConvSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_sum",
            prec=0.1)


    def test_output_conv_sum_3d(self):
        self._test_output(
            ConvSum(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_sum")
        self._test_output_bf16(
            ConvSum(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_sum",
            prec=0.04)


    def test_output_cascaded_conv_bn_sum_relu_2d(self):
        self._test_output(
            CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_sum_relu",
            kind_not_in_graph="aten::batch_norm")
        self._test_output_bf16(
            CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 64, 64),
            kind_in_graph="ipex::conv2d_sum_relu",
            kind_not_in_graph="aten::batch_norm",
            prec=0.02)


    def test_output_cascaded_conv_bn_sum_relu_3d(self):
        self._test_output(
            CascadedConvBnSumRelu(3, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_sum_relu",
            kind_not_in_graph="aten::batch_norm",)
        self._test_output_bf16(
            CascadedConvBnSumRelu(3, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 32, 32, 32),
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
            kind_in_graph="ipex::linear_relu",
            prec=0.02)
        self._test_output(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_relu")
        self._test_output_bf16(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_relu",
            prec=0.02)
 
    def test_output_linear_add(self):
        self._test_output(
            LinearAdd(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")

    def test_output_linear_reshape_relu(self):
        self._test_output(
            Linear_Reshape_Relu(3, 32,(64,16),bias=True),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")

    def test_output_linear_sigmoid(self):
        self._test_output(
            LinearSigmoid(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")

    def test_output_linear_bn(self):
        self._test_output(
            LinearBn(2 ,32, 32, bias=True),
            torch.rand(1, 1, 32, 32),
            kind_in_graph="aten::linear")

    def test_output_linear_reshape_bn(self):
        self._test_output(
            Linear_Reshape_Bn(2 ,32, 32,(1,1,64,16),bias=True),
            torch.rand(1, 1, 32, 32),
            kind_in_graph="aten::linear")

    def test_output_linear_gelu(self):
        self._test_output(
            LinearGelu(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_gelu")
        # self._test_output_bf16(
        #     LinearGelu(3, 32, bias=True),
        #     torch.rand(32, 3),
        #     kind_in_graph="ipex::linear_gelu",
        #     prec=5e-3)
        self._test_output(
            LinearGelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex::linear_gelu")
        # self._test_output_bf16(
        #     LinearGelu(3, 32, bias=False),
        #     torch.rand(32, 3),
        #     kind_in_graph="ipex::linear_gelu",
        #     prec=5e-3)


    # def test_channel_shuffle(self):
    #     self._test_output(
    #         ChannelShuffle(10, 16, 50, 50, 4),
    #         torch.rand(10, 16, 50, 50),
    #         kind_in_graph="ipex::shuffle_2d")


    def test_jit_function(self):
        # test hool trace and script can works for function
        def fn(input, weight, bias):
            return F.linear(input, weight, bias)

        input = torch.randn(2, 4)
        weight = torch.randn(5, 4)
        bias = torch.randn(5)
        result = fn(input, weight, bias)

        scripted_fn = torch.jit.script(fn)
        traced_fn = torch.jit.trace(fn, (input, weight, bias))

        self.assertEqual(scripted_fn(input, weight, bias), result)
        self.assertEqual(traced_fn(input, weight, bias), result)


    def test_jit_conv_sum_in_diff_block(self):
        self._test_output(
            ConvSumInDiffBlock(2, 3, 32, kernel_size=1, stride=1, padding=0),
            torch.rand(32, 3, 64, 64),
            kind_not_in_graph="ipex::conv2d_sum")
        self._test_output_bf16(
            ConvSumInDiffBlock(2, 3, 32, kernel_size=1, stride=1, padding=0),
            torch.rand(32, 3, 64, 64),
            kind_not_in_graph="ipex::conv2d_sum")

    # def test_manmually_fused_linear_relu(self):
    #     m =  LinearRelu(3, 32, bias=True).eval()
    #     x = torch.rand(32, 3)
    #     with torch.no_grad():
    #         result = m(x)
    #     fused_m = ipex.LinearRelu(3, 32)
    #     fused_m.weight = m.linear.weight
    #     fused_m.bias = m.linear.bias
    #     with torch.no_grad():
    #         fused_result = fused_m(x)
    #     self.assertEqual(fused_result, result)


if __name__ == '__main__':
    torch.manual_seed(2020)
    # core.enable_auto_dnnl()
    test = unittest.main()