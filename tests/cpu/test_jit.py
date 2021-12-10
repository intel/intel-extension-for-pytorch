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
import warnings
import itertools

import torch
import torch.nn as nn
from torch.jit._recursive import wrap_cpp_module
import torch.fx.experimental.optimization as optimization
import copy

import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch._six import inf, nan
import re

from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, brute_pdist, brute_cdist, slowTest, \
    skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf, skipIfSpecificVersions

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
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        return torch.add(F.relu(self.conv1(x), inplace=True),self.conv2(x))

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

class ConvScalarSum(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvScalarSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        b = self.conv(x)
        return b+2

class ConvBroadcastSum(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBroadcastSum, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv1(x)
        return a[1:2].clone()+b

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
        self.linear1 = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x1 = x.clone()
        return torch.add(self.linear(x),self.linear1(x1))

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

class ConvSiluOutplace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSiluOutplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)
        self.silu = nn.SiLU()

    def forward(self, x):
        a1 = self.conv2d(x)
        b1 = self.silu(a1)
        return b1

class ConvSiluInplace(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size):
        super(ConvSiluInplace, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, image_size)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        a1 = self.conv2d(x)
        b1 = self.silu(a1)
        return b1

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

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2d, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, x):
        x = self.conv_transpose2d(x)
        return x

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

class MatmulDiv(nn.Module):
    def __init__(self, div_scalar=False, with_out=False):
        super(MatmulDiv, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        mm_res = None
        y = torch.transpose(x, 1, 2).contiguous()
        mm_res_shape = x.size()[:-1] + (y.size()[-1:])
        if self.with_out:
            mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
            torch.matmul(x, y, out=mm_res)
        else:
            mm_res = torch.matmul(x, y)
        if self.div_scalar:
            return mm_res.div(2.0)
        else:
            return mm_res.div(torch.ones(mm_res_shape,dtype=x.dtype)+1)

class MHAScoresCalculation(nn.Module):
    def __init__(self, dim_per_head, softmax_dim=-1):
        super(MHAScoresCalculation, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, bias):
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        scores = qk + bias
        return self.softmax(scores)

class AtenSoftmaxRepalce(nn.Module):
    def __init__(self, dim=-1):
        super(AtenSoftmaxRepalce, self).__init__()
        self.softmax = torch.nn.Softmax(dim)

    def forward(self, x):
        return self.softmax(x)

class AtenBatchNormRepalce(nn.Module):
    def __init__(self):
        super(AtenBatchNormRepalce, self).__init__()
        self.bn = torch.nn.BatchNorm2d(10)

    def forward(self, x):
        return self.bn(x)

class AddLayerNorm(torch.nn.Module):
    def __init__(self, dim=32):
        super(AddLayerNorm, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)
    def forward(self, x, y):
        x = torch.add(x,y)
        return self.layernorm(x)

class AddLayerNorm_v1(torch.nn.Module):
    def __init__(self, dim=32):
        super(AddLayerNorm_v1, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)
    def forward(self, x, y, z):
        x = x + y + z
        return self.layernorm(x)

class ModMultLinear(nn.Module):
    def __init__(self, w1_dim, w2_dim):
         super(ModMultLinear, self).__init__()
         self.linear1 = nn.Linear(5, w1_dim)
         self.linear2 = nn.Linear(5, w2_dim)
         self.linear3 = nn.Linear(w1_dim, 5)
         self.linear4 = nn.Linear(w1_dim, 5)

    def forward(self, x):
         res1 = self.linear1(x)
         res2 = self.linear2(x)
         res3 = self.linear3(res1)
         res4 = self.linear4(res1)
         return res1, res2, res3, res4

class Tester(TestCase):

    def _test_output(self, model, x, kind_in_graph=None, kind_not_in_graph=None, levels=['O0','O1']):
        modelName = model.__class__.__name__
        for level in levels:
            ipex.enable_onednn_fusion(False)
            model = model.eval()
            # It will be removed after jit support conv_bn folding
            if level == 'O0':
                try:
                    model = optimization.fuse(model)
                except:
                    warnings.warn("Conv BatchNorm folding failed.")
            if x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)
                model = model.to(memory_format=torch.channels_last)
            model = ipex.optimize(model, dtype=torch.float32, level=level)
            with torch.no_grad():
                result = model(x)
                traced_model = torch.jit.trace(model, x).eval()
                traced_model = torch.jit.freeze(traced_model)
                tresult = traced_model(x)

            self.assertEqual(result, tresult)

            ipex.enable_onednn_fusion(True)
            with torch.no_grad():
                trace_fused_model = torch.jit.trace(model, x)
                trace_fused_model = torch.jit.freeze(trace_fused_model)

                # enable fusiong in ipex.
                fused_tresult = trace_fused_model(x)
                # conv relu fusion, conv sum fusion or conv sum relu fusion
                trace_graph = trace_fused_model.graph_for(x)
                fused_tresult = trace_fused_model(x)
            self.assertEqual(result, fused_tresult)
            # check if the fused node exists in the graph
            if kind_in_graph is not None:
                self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

            # check if certain node does not exist in the graph
            if kind_not_in_graph is not None:
                self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))


    def _test_output_bf16(self, model, x, kind_in_graph=None, kind_not_in_graph=None, prec=None, levels=['O0','O1']):
        modelName = model.__class__.__name__
        for level in levels:
            ipex.enable_onednn_fusion(True)
            model = model.eval()
            # It will be removed after jit support conv_bn folding
            if level == 'O0':
                try:
                    model = optimization.fuse(model)
                except:
                    warnings.warn("Conv BatchNorm folding failed.")
            if x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)
                model = model.to(memory_format=torch.channels_last)
            model = ipex.optimize(model, dtype=torch.bfloat16, level=level)
            x2 = x.clone()
            x3 = x.clone()

            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
                # bf16, native path
                result = model(x)
                trace_fused_model = torch.jit.trace(copy.deepcopy(model), x3)
                trace_fused_model = torch.jit.freeze(trace_fused_model)
                # enable fusion path.
                fused_tresult = trace_fused_model(x3)
                # bf16, jit trace path
                trace_graph = trace_fused_model.graph_for(x3)
                fused_tresult = trace_fused_model(x3)

            self.assertEqual(fused_tresult, result, prec=prec)
            self.assertEqual(fused_tresult.dtype, torch.bfloat16)

            # check if the fused node exists in the graph
            if kind_in_graph is not None:
                self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

            # check if certain node does not exist in the graph
            if kind_not_in_graph is not None:
                self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))

    def test_jit_freeze(self):
        model = ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1).eval()
        x = torch.randn(32, 3, 64, 64).to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model, dtype=torch.float32)

        with torch.no_grad():
            trace_model = torch.jit.trace(model, x).eval()

        freeze_model = torch.jit.freeze(trace_model)
        with torch.no_grad():
            # enable fusiong in ipex.
            result1 = trace_model(x)
            result2 = freeze_model(x)
            # conv relu fusion, conv sum fusion or conv sum relu fusion
            trace_graph = trace_model.graph_for(x)
            freeze_graph = freeze_model.graph_for(x)

        node = "ipex_prepack::convolution_prepack"
        # prepack op need in freeze model
        self.assertTrue(all(n.kind() != node for n in freeze_graph.nodes()))
        #  prepack op need note in none freeze model
        self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))

    def test_concat_linear(self):
        def check_op_count(graph_str, op_names=[]):
            count = 0
            node_list = graph_str.strip().split("\n")
            for node in node_list:
                for op_name in op_names:
                  if op_name in node:
                    count += 1
            return count
        origin_model = ModMultLinear(50, 60).eval()

        test_val1 = torch.rand([50, 5])
        # call mkl path(fp32)
        model = ipex.optimize(origin_model, dtype=torch.float32)
        ori_res = model(test_val1)
        model_jit = torch.jit.trace(model,(test_val1))
        graph_ori = str(model_jit.graph_for(test_val1))
        linear_count_ori = check_op_count(graph_ori, ["aten::linear"])
        self.assertEqual(linear_count_ori, 4)
        model_jit = torch.jit.freeze(model_jit)
        jit_res = model_jit(test_val1)
        self.assertEqual(ori_res, jit_res)
        graph_opt = str(model_jit.graph_for(test_val1))
        linear_count_ori = check_op_count(graph_opt, ["aten::linear"])
        self.assertEqual(linear_count_ori, 2)
        # call onednn path(fp32)
        model = ipex.optimize(origin_model, dtype=torch.float32, auto_kernel_selection=True)
        ori_res = model(test_val1)
        model_jit = torch.jit.trace(model,(test_val1))
        graph_ori = str(model_jit.graph_for(test_val1))
        linear_count_ori = check_op_count(graph_ori, ["ipex_prepack::linear_run"])
        self.assertEqual(linear_count_ori, 4)
        model_jit = torch.jit.freeze(model_jit)
        jit_res = model_jit(test_val1)
        self.assertEqual(ori_res, jit_res)
        graph_opt = str(model_jit.graph_for(test_val1))
        linear_count_ori = check_op_count(graph_opt, ["ipex_prepack::linear_run"])
        self.assertEqual(linear_count_ori, 2)

        model = ipex.optimize(origin_model, dtype=torch.bfloat16)
        test_val1 = test_val1.bfloat16()
        with torch.cpu.amp.autocast(), torch.no_grad():
            ori_res = model(test_val1)
            model_jit = torch.jit.trace(model,(test_val1))
            graph_ori = str(model_jit.graph_for(test_val1))
            linear_count_ori = check_op_count(graph_ori, ["ipex_prepack::linear_run"])
            self.assertEqual(linear_count_ori, 4)
            model_jit = torch.jit.freeze(model_jit)
            model_jit(test_val1)
            graph_opt = str(model_jit.graph_for(test_val1))
            jit_res = model_jit(test_val1)
            self.assertEqual(ori_res[1], jit_res[1])
            linear_count_ori = check_op_count(graph_opt, ["ipex_prepack::linear_run"])
            self.assertEqual(linear_count_ori, 2)

    def test_add_layernorm(self):
        bs = 56
        seq_len = 384
        dim = 768
        a = torch.randn(bs, seq_len, dim)
        b = torch.randn(bs, seq_len, dim)
        model = AddLayerNorm(dim)
        jit_model = torch.jit.trace(model,(a, b))
        trace_graph = jit_model.graph_for(a, b)
        jit_res = jit_model(a, b)
        ori_res = model(a, b)
        self.assertEqual(jit_res, ori_res)
        node = "ipex::add_layernorm"
        self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))

        a_bf16 = a.to(torch.bfloat16)
        b_bf16 = b.to(torch.bfloat16)
        with torch.cpu.amp.autocast():
            ori_res = model(a_bf16, b_bf16)
            model_jit = jit_model = torch.jit.trace(model,(a, b))
            trace_graph = jit_model.graph_for(a, b)
            jit_res = jit_model(a_bf16, b_bf16)
            node = "ipex::add_layernorm"
            self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))
            self.assertEqual(jit_res, ori_res, prec=5e-2)

        model = AddLayerNorm_v1(dim)
        c = torch.randn(bs, seq_len, dim)
        jit_model = torch.jit.trace(model,(a, b, c))
        trace_graph = jit_model.graph_for(a, b, c)
        jit_res = jit_model(a, b, c)
        ori_res = model(a, b, c)
        self.assertEqual(jit_res, ori_res)
        node = "ipex::add_layernorm"
        self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))

    def test_mha_scores_calculation(self):
        def _test_pure_bf16(model, trace_model, mat1, mat2, bias, prec=3e-2):
            mat1_bf16 = mat1.to(torch.bfloat16)
            mat2_bf16 = mat2.to(torch.bfloat16)
            bias_bf16 = bias.to(torch.bfloat16)
            res_ref = model(mat1_bf16, mat2_bf16, bias_bf16)
            res_jit = trace_model(mat1_bf16, mat2_bf16, bias_bf16)
            self.assertEqual(res_ref, res_jit, prec=prec)

        mat1 = torch.randn(56, 16, 384, 384)
        mat2 = torch.randn(56, 16, 384, 384)
        bias = torch.randn(56, 16, 384, 384)

        for softmax_dim in [0, 1, 2, -1]:
            mha = MHAScoresCalculation(4, softmax_dim)
            with torch.no_grad():
                mha_jit = torch.jit.trace(mha, (mat1, mat2, bias))
                mha_jit.eval()

                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(1, 1, 2, 3)
                mat2 = torch.randn(1, 1, 16, 3)
                bias = torch.randn(1, 1, 2, 16)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(1, 1, 2, 3)
                mat2 = torch.randn(1, 1, 32, 3)
                bias = torch.randn(1, 1, 2, 32)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(1, 1, 2, 3)
                mat2 = torch.randn(1, 1, 33, 3)
                bias = torch.randn(1, 1, 2, 33)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(2, 3, 4, 6)
                mat2 = torch.randn(2, 3, 6, 6)
                bias = torch.randn(2, 3, 4, 6)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                # Test broadcast
                mat1 = torch.randn(2, 3, 4, 10)
                mat2 = torch.randn(2, 3, 16, 10)
                bias = torch.randn(1, 1, 1, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(4, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(3, 1, 1)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(2, 1, 1, 1)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(3, 4, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(2, 1, 1, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(2, 1, 4, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

    def test_conv2d_fusion(self):
        batch_size = 32
        out_channels = 64
        in_channels = 3
        kernel_size = 3
        image_size = 64
        '''
        self._test_output(
            ConvSwishOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run")
        '''
        self._test_output_bf16(
            ConvSwishOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run",
            prec=0.02)
        self._test_output(
            ConvSwishInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run")
        self._test_output_bf16(
            ConvSwishInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run",
            prec=0.05)
        self._test_output(
            ConvSiluOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run")
        self._test_output(
            ConvSiluInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run")
        self._test_output_bf16(
            ConvSiluOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run",
            prec=0.02)
        self._test_output_bf16(
            ConvSiluInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_swish_run",
            prec=0.05)
        self._test_output(
            ConvSigmoidOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_sigmoid_run")
        self._test_output_bf16(
            ConvSigmoidOutplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_sigmoid_run",
            prec=0.02)
        self._test_output(
            ConvSigmoidInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_sigmoid_run")
        self._test_output_bf16(
            ConvSigmoidInplace(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_sigmoid_run",
            prec=0.02)
        self._test_output(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size, True),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_hardtanh_run")
        self._test_output_bf16(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size, True),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_hardtanh_run",
            prec=0.05)
        self._test_output(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_hardtanh_run")
        self._test_output_bf16(
            ConvHardtanh(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_hardtanh_run",
            prec=0.05)
        self._test_output(
            ConvElu(in_channels, out_channels, kernel_size, image_size, True),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_elu_run")
        # self._test_output_bf16(
        #     ConvElu(in_channels, out_channels, kernel_size, image_size, True),
        #     torch.randn(batch_size, in_channels, image_size, image_size),
        #     kind_in_graph="ipex::conv2d_elu",
        #     prec=0.02)
        self._test_output(
            ConvElu(in_channels, out_channels, kernel_size, image_size),
            torch.randn(batch_size, in_channels, image_size, image_size),
            kind_in_graph="ipex_prepack::convolution_elu_run")
        # self._test_output_bf16(
        #     ConvElu(in_channels, out_channels, kernel_size, image_size),
        #     torch.randn(batch_size, in_channels, image_size, image_size),
        #     kind_in_graph="ipex::conv2d_elu",
        #     prec=0.02)

    def test_output_conv_bn_2d(self):
        self._test_output(
            ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex::batch_norm",
            levels=['O1'])
        self._test_output_bf16(
            ConvBatchNorm_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex::batch_norm",
            prec=0.02,
            levels=['O1'])

    def test_output_bn_conv_2d(self):
        self._test_output(
            BatchNormConv_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::batch_norm",
            kind_not_in_graph=None)

    def test_output_bn_conv_bn(self):
        self._test_output(
            BatchNorm_Conv_BatchNorm(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::batch_norm",
            kind_not_in_graph=None)

    def test_output_conv_reshape_bn_2d(self):
        self._test_output(
            ConvReshapeBatchNorm(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex::batch_norm",
            kind_not_in_graph=None)

    def test_output_conv_conv_concate(self):
        self._test_output(
            Conv_Conv_Concat(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph=None)

    def test_output_conv_relu_add(self):
        self._test_output(
            Conv_Relu_Add(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_relu_run")

    def test_output_conv_bn_relu(self):
        self._test_output(
            Conv_Bn_Relu(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_relu_run")

    def test_output_conv_reshape_relu(self):
        self._test_output(
            ConvReshapeRelu(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_relu_run")

    def test_output_conv_reshape_sum(self):
        self._test_output(
            ConvReshapeSum(2, 3, 32, (64, 16, 62, 62), kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_add_run")

    def test_output_conv_bn_3d(self):
        self._test_output(
            ConvBatchNorm_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="aten::conv3d",
            kind_not_in_graph="ipex::batch_norm")

    def test_output_conv_relu_2d(self):
        self._test_output(
            ConvRelu_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_relu_run")
        self._test_output_bf16(
            ConvRelu_Fixed(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_relu_run",
            prec=0.02)

    def test_output_conv_relu_3d(self):
        self._test_output(
            ConvRelu_Fixed(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_relu")

    def test_output_conv_sum_2d(self):
        self._test_output(
            ConvSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_add_run")
        self._test_output_bf16(
            ConvSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_add_run",
            prec=0.1)

    def test_output_conv_sum_3d(self):
        self._test_output(
            ConvSum(3, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_sum")

    def test_output_conv_scalar_sum_2d(self):
        self._test_output(
            ConvScalarSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_add_run")
        self._test_output_bf16(
            ConvScalarSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_add_run",
            prec=0.1)

    def test_output_conv_broadcast_sum_2d(self):
        self._test_output(
            ConvBroadcastSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_add_run")
        self._test_output_bf16(
            ConvBroadcastSum(2, 3, 32, kernel_size=3, stride=1),
            torch.randn(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_add_run",
            prec=0.1)

    def test_output_cascaded_conv_bn_sum_relu_2d(self):
        self._test_output(
            CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_add_relu_run",
            kind_not_in_graph="ipex::batch_norm")
        self._test_output_bf16(
            CascadedConvBnSumRelu(2, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 64, 64),
            kind_in_graph="ipex_prepack::convolution_add_relu_run",
            kind_not_in_graph="ipex::batch_norm",
            prec=0.02)

    def test_output_cascaded_conv_bn_sum_relu_3d(self):
        self._test_output(
            CascadedConvBnSumRelu(3, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_sum_relu",
            kind_not_in_graph="ipex::batch_norm")
        self._test_output_bf16(
            CascadedConvBnSumRelu(3, 3, 64, 32, kernel_size=3, stride=1),
            torch.rand(32, 3, 32, 32, 32),
            kind_in_graph="ipex::conv3d_sum_relu",
            kind_not_in_graph="ipex::batch_norm",
            prec=0.02)

    def test_output_conv_transpose2d(self):
        def _deconv_params_list():
            params_dict = {
                "input_height": [12],
                "input_width": [12],
                "input_depth": [12],
                "input_channel_per_group": [15],
                "output_channel_per_group": [3],
                "kernel_size": [3],
                "bias": [True, False],
                "stride": [1, 2],
                "padding": [1, 2],
                "output_padding": [0],  # TODO: fix output_padding  >1.
                "groups": [1, 2],
                "dilation": [1, 2],
            }

            params_list = []

            for key, value in params_dict.items():
                params_list.append(value)
            return params_list

        params_list = _deconv_params_list()

        for input_width, input_height, input_depth, input_channel_per_group, output_channel_per_group, kernel_size, bias, stride, padding, output_padding, groups, dilation in itertools.product(*params_list):
            if (output_padding < stride or output_padding < dilation) \
                    and ((input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0) \
                    and ((input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0) \
                    and ((input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0):

                ic = input_channel_per_group * groups
                oc = output_channel_per_group * groups

                x = torch.randn(2, ic, input_height, input_width)
                model = ConvTranspose2d(ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation)

                self._test_output(
                    model,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose2d_run",
                    kind_not_in_graph="aten::conv_transpose2d",
                    levels=["O0"])
                if not re.findall('[\d\.]*', torch.__version__)[0] in ['1.10.0',]:
                    self._test_output_bf16(
                        model,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose2d_run",
                        kind_not_in_graph="aten::conv_transpose2d",
                        levels=["O0"],
                        prec=0.02)
                self._test_output(
                    model,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose2d_run",
                    kind_not_in_graph="torch_ipex::conv_transpose2d",
                    levels=["O1"])
                self._test_output_bf16(
                    model,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose2d_run",
                    kind_not_in_graph="torch_ipex::conv_transpose2d",
                    levels=["O1"],
                    prec=0.02)

    def test_linear_auto_kernel_selection_fp32(self):
        x = torch.rand(32, 3)
        options = itertools.product(['O0', 'O1'], [True, False])
        for level, auto_select_kernel in options:
            model = LinearRelu(3, 32, bias=True).eval()
            model = ipex.optimize(model, dtype=torch.float32, level=level, auto_kernel_selection=auto_select_kernel)
            with torch.no_grad():
                traced_model = torch.jit.trace(model, x).eval()
                traced_model = torch.jit.freeze(traced_model)
                y = traced_model(x)
                trace_graph = traced_model.graph_for(x)

                if auto_select_kernel and level == 'O1':
                    # for 'O1' and auto_select_kernel is True, we will use ipex linear
                    self.assertTrue(any(n.kind() == 'ipex_prepack::linear_relu_run' for n in trace_graph.nodes()))
                else:
                    # for 'O1' and auto_select_kernel is false or 'O0', we will use mkl linear
                    self.assertTrue(any(n.kind() == 'aten::linear' for n in trace_graph.nodes()))

    def test_linear_auto_kernel_selection_bf16(self):
        x = torch.rand(32, 3)
        options = itertools.product(['O0', 'O1'], [True, False])
        for level, auto_select_kernel in options:
            model = LinearRelu(3, 32, bias=True).eval()
            model = ipex.optimize(model, dtype=torch.bfloat16, level=level, auto_kernel_selection=auto_select_kernel)
            with torch.cpu.amp.autocast(), torch.no_grad():
                traced_model = torch.jit.trace(model, x).eval()
                traced_model = torch.jit.freeze(traced_model)
                y = traced_model(x)
                trace_graph = traced_model.graph_for(x)

                # for bfloat16 path, we will use ipex linear for 'O0' and 'O1'
                self.assertTrue(any(n.kind() == 'ipex_prepack::linear_relu_run' for n in trace_graph.nodes()))

    def test_output_linear_relu(self):
        self._test_output(
            LinearRelu(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")
        self._test_output_bf16(
            LinearRelu(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_relu_run",
            prec=0.02)
        self._test_output(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")
        self._test_output_bf16(
            LinearRelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_relu_run",
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
        self._test_output_bf16(
            LinearSigmoid(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_run",
            prec=0.02)

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
            kind_in_graph="aten::linear")
        self._test_output_bf16(
            LinearGelu(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_gelu_run",
            prec=5e-3)
        self._test_output(
            LinearGelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")
        self._test_output_bf16(
            LinearGelu(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_gelu_run",
            prec=5e-3)

    def test_channel_shuffle(self):
        self._test_output(
            ChannelShuffle(10, 16, 50, 50, 4),
            torch.rand(10, 16, 50, 50),
            kind_in_graph="ipex::shuffle_2d")

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
            kind_not_in_graph="ipex_prepack::convolution_add_run")
        self._test_output_bf16(
            ConvSumInDiffBlock(2, 3, 32, kernel_size=1, stride=1, padding=0),
            torch.rand(32, 3, 64, 64),
            kind_not_in_graph="ipex_prepack::convolution_add_run")

    def test_matmul_div(self):
        self._test_output(
            MatmulDiv(div_scalar=True, with_out=True),
            torch.randn(10, 3, 4),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None)
        self._test_output(
            MatmulDiv(div_scalar=True, with_out=False),
            torch.randn(10, 3, 4),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None)
        self._test_output(
            MatmulDiv(div_scalar=False, with_out=False),
            torch.randn(10, 3, 4),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None)
        self._test_output(
            MatmulDiv(div_scalar=False, with_out=True),
            torch.randn(10, 3, 4),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None)
        self._test_output_bf16(
            MatmulDiv(div_scalar=True, with_out=True),
            torch.randn(10, 3, 4, dtype=torch.bfloat16),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None,
            prec=5e-2)
        self._test_output_bf16(
            MatmulDiv(div_scalar=True, with_out=False),
            torch.randn(10, 3, 4, dtype=torch.bfloat16),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None,
            prec=5e-2)
        self._test_output_bf16(
            MatmulDiv(div_scalar=False, with_out=True),
            torch.randn(10, 3, 4, dtype=torch.bfloat16),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None,
            prec=5e-3)
        self._test_output_bf16(
            MatmulDiv(div_scalar=False, with_out=False),
            torch.randn(10, 3, 4, dtype=torch.bfloat16),
            kind_in_graph="ipex::matmul_div",
            kind_not_in_graph=None,
            prec=5e-3)

    def test_ipex_softmax(self):
        self._test_output(
            AtenSoftmaxRepalce(),
            torch.rand(3, 4, 4),
            kind_in_graph="ipex::softmax")
        self._test_output_bf16(
            AtenSoftmaxRepalce(),
            torch.rand(3, 4, 4, dtype=torch.bfloat16),
            kind_in_graph="ipex::softmax",
            prec=5e-3)

    def test_ipex_batch_norm(self):
        self._test_output(
            AtenBatchNormRepalce(),
            torch.rand(10, 10, 4, 4),
            kind_in_graph="ipex::batch_norm")
        self._test_output_bf16(
            AtenBatchNormRepalce(),
            torch.rand(10, 10, 4, 4, dtype=torch.bfloat16),
            kind_in_graph="ipex::batch_norm",
            prec=5e-3)

    def test_restore_inplace(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, params_dict={}):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3, 3)
                self.eltwise = eltwise_fn
                self.params_dict = params_dict

            def forward(self, x):
                x = x * 3.1
                x = self.eltwise(x, **self.params_dict)
                x = self.conv(x)
                return x

        for eltwise in ['sigmoid', 'tanh', 'celu', 'elu', 'hardsigmoid', 'hardswish', 'hardtanh', 'leaky_relu', 'relu6', 'relu', 'rrelu', 'selu', 'silu', 'clamp']:
            eltwise_fn_name = eltwise + '_'
            if eltwise in ['sigmoid', 'tanh', 'celu', 'relu', 'rrelu', 'selu']:
                # use torch.sigmoid_(x)
                eltwise_fn = getattr(torch, eltwise_fn_name)
                m = M(eltwise_fn)
            elif eltwise == 'clamp':
                eltwise_fn = getattr(torch, eltwise_fn_name)
                m = M(eltwise_fn, {"min": 0, "max": 2})
            else:
                # use F.elu(x, inplace=True)
                eltwise_fn = getattr(F, eltwise)
                m = M(eltwise_fn, {"inplace": True})

            with torch.no_grad():
                m.eval()
                x = torch.randn(1, 3, 16, 16)
                traced = torch.jit.trace(m, x)
                trace_graph = traced.graph_for(x)
                self.assertTrue(any(n.kind() == "aten::" + eltwise_fn_name for n in trace_graph.nodes()))

                y = m(x)
                traced_y = traced(x)
                self.assertEqual(y, traced_y)

if __name__ == '__main__':
    torch.manual_seed(2020)
    test = unittest.main()
