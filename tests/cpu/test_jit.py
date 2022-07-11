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
import time
import sys
import warnings
import itertools
import contextlib
import torch
import torch.nn as nn
import torch.fx.experimental.optimization as optimization
import copy

import intel_extension_for_pytorch as ipex
import intel_extension_for_pytorch._C as core

import torch.nn as nn
import torch.nn.functional as F

from common_utils import TestCase

def get_rand_seed():
    return int(time.time() * 1000000000)

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

device = 'cpu:0'
SIZE = 100

conv_module = {1: torch.nn.Conv1d, 2 : torch.nn.Conv2d, 3 : torch.nn.Conv3d}
convtranspose_module = {2 : torch.nn.ConvTranspose2d, 3 : torch.nn.ConvTranspose3d}
bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}

class EltwiseFusionOp:
    def __init__(self, ipex_eltwise_op, bf16_supported=True, prec=0.02, op_input_list={}):
        self.ipex_eltwise_op = ipex_eltwise_op
        self.bf16_supported = bf16_supported
        self.prec = prec
        self.op_input_list = op_input_list

unary_PyTorch_op_to_IPEX_op_map = {
    # PyTorch_op_name: [ipex_op_name, BF16_supported, prec]
    torch.relu: EltwiseFusionOp("relu"),
    torch.relu_: EltwiseFusionOp("relu"),
    torch.sigmoid: EltwiseFusionOp("sigmoid"),
    torch.sigmoid_: EltwiseFusionOp("sigmoid"),
    nn.SiLU(inplace=True): EltwiseFusionOp("swish"),
    nn.SiLU(inplace=False): EltwiseFusionOp("swish"),
    torch.tanh: EltwiseFusionOp("tanh"),
    torch.tanh_: EltwiseFusionOp("tanh"),
    nn.Mish(inplace=True): EltwiseFusionOp("mish", bf16_supported=False), # TODO: support bf16 mish_ in stock PyTorch
    nn.Mish(inplace=False): EltwiseFusionOp("mish", bf16_supported=False), # TODO: support bf16 mish in stock PyTorch
    torch.abs: EltwiseFusionOp("abs"),
    torch.abs_: EltwiseFusionOp("abs"),
    torch.exp: EltwiseFusionOp("exp", prec=0.035),
    torch.exp_: EltwiseFusionOp("exp", prec=0.035),
    torch.nn.Hardswish(inplace=True): EltwiseFusionOp("hardswish"),
    torch.nn.Hardswish(inplace=False): EltwiseFusionOp("hardswish"),
    torch.square: EltwiseFusionOp("square", prec=0.035),
    torch.square_: EltwiseFusionOp("square", prec=0.035),
    torch.nn.Hardsigmoid(inplace=True): EltwiseFusionOp("hardsigmoid"),
    torch.nn.Hardsigmoid(inplace=False): EltwiseFusionOp("hardsigmoid"),
}

non_unary_PyTorch_op_to_IPEX_op_map = {
    nn.GELU(approximate="none"): EltwiseFusionOp("gelu"),
    nn.GELU(approximate="tanh"): EltwiseFusionOp("gelu"),
    nn.LeakyReLU(0.1, inplace=True): EltwiseFusionOp("leaky_relu"),
    nn.LeakyReLU(0.1, inplace=False): EltwiseFusionOp("leaky_relu"),
    nn.Hardtanh(inplace=True): EltwiseFusionOp("hardtanh"),
    nn.Hardtanh(inplace=False): EltwiseFusionOp("hardtanh"),
    nn.ELU(inplace=True): EltwiseFusionOp("elu"),
    nn.ELU(inplace=False): EltwiseFusionOp("elu"),
    torch.clamp: EltwiseFusionOp("hardtanh", op_input_list={"min": -2, "max": 3}),
    torch.clamp_: EltwiseFusionOp("hardtanh", op_input_list={"min": -2, "max": 3}),
    torch.pow: EltwiseFusionOp("pow", op_input_list={"exponent": 2}),
    lambda t: t.pow_(2): EltwiseFusionOp("pow"),
}

unsupported_PyTorch_op_to_IPEX_op_map = {
    torch.clamp: EltwiseFusionOp("hardtanh", op_input_list={"min": -2}), # clamp fusion requires that neither of min and max is None
    torch.clamp_: EltwiseFusionOp("hardtanh", op_input_list={"max": 3}), # clamp_ fusion requires that neither of min and max is None
    torch.pow: EltwiseFusionOp("pow", op_input_list={"exponent": torch.randn(1)}), # pow fusion requires exponent to be a Scalar but not a Tensor
    lambda t: t.pow_(torch.randn(1)): EltwiseFusionOp("pow"), # pow_ fusion requires exponent to be a Scalar but not a Tensor
}

# The below eltwise OP have unstable numeric issue.
# We will run the tests with fixed seed to avoid false positive.
# For example, for log, when running bf16 linear-log test
# y = linear(x)
# z = log(y)
# Supposing we meet a case where
# y_fp32 = 0 and y_bf16 = 0.0008
# Then z_fp32 = log(0) = nan
#      z_bf16 = log(0.0008) = -7.1309
# We're not able to directly compare z_fp32 with z_bf16.
PyTorch_op_to_IPEX_op_fixed_seed_map = {
    torch.log: EltwiseFusionOp("log", prec=0.065),
    torch.log_: EltwiseFusionOp("log", prec=0.065),
    torch.round: EltwiseFusionOp("round"),
    torch.round_: EltwiseFusionOp("round"),
    torch.sqrt: EltwiseFusionOp("sqrt"),
    torch.sqrt_: EltwiseFusionOp("sqrt"),
}

class ConvEltwise(nn.Module):
    def __init__(self, eltwise_fn, dim, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvEltwise, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.conv(x)
        a = a / 2
        b = self.eltwise(a, **self.kwargs)
        return b

class LinearEltwise(nn.Module):
    def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
        super(LinearEltwise, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.linear(x)
        a = a / 2
        b = self.eltwise(a, **self.kwargs)
        return b

class ConvTransposeEltwise(nn.Module):
    def __init__(self, eltwise_fn, dim, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeEltwise, self).__init__()
        self.conv_transpose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.eltwise = eltwise_fn
        self.kwargs = kwargs

    def forward(self, x):
        a = self.conv_transpose(x)
        b = self.eltwise(a, **self.kwargs)
        return b

class ConvTransposeSumAccumuOnRight(nn.Module):
    def __init__(self, dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeSumAccumuOnRight, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.convtranspose1 = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.add_func = add_func
        self.kwargs = kwargs

    def forward(self, x):
        a = self.convtranspose(x)
        b = F.relu(self.convtranspose1(x))
        return self.add_func(a, b, self.kwargs)

class ConvTransposeSumAccumuOnLeft(nn.Module):
    def __init__(self, dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeSumAccumuOnLeft, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.convtranspose1 = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.add_func = add_func
        self.kwargs = kwargs

    def forward(self, x):
        a = F.relu(self.convtranspose(x))
        b = self.convtranspose1(x)
        return self.add_func(a, b, self.kwargs)

class ConvTransposeSumBroadcast(nn.Module):
    def __init__(self, dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs):
        super(ConvTransposeSumBroadcast, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, 1, kernel_size, image_size)
        self.convtranspose1 = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.add_func = add_func
        self.kwargs = kwargs

    def forward(self, x):
        a = F.relu(self.convtranspose(x))
        b = self.convtranspose1(x)
        return self.add_func(a, b, self.kwargs)

class ConvTransposeAddRelu(nn.Module):
    def __init__(self, dim, in_channels, mid_channels, out_channels, kernel_size, inplace, **kwargs):
        super(ConvTransposeAddRelu, self).__init__()
        self.convtranspose = convtranspose_module[dim](in_channels, mid_channels, kernel_size, padding=1, bias=False, **kwargs)
        self.convtranspose1 = convtranspose_module[dim](
            mid_channels, out_channels, kernel_size, padding=1, bias=False, **kwargs)
        self.convtranspose2 = convtranspose_module[dim](in_channels, out_channels, kernel_size, padding=1, bias=False, **kwargs)
        self.inplace = inplace

    def forward(self, x):
        a = self.convtranspose(x)
        a = F.relu(a, inplace=self.inplace)
        a = self.convtranspose1(a)
        b = self.convtranspose2(x)
        return F.relu(a.add_(b), inplace=self.inplace)

class ConvBatchNorm_Fixed(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBatchNorm_Fixed2(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001, track_running_stats=False)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBatchNorm_Fixed3(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvBatchNorm_Fixed3, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=True, **kwargs)
        self.bn = bn_module[dim](out_channels, eps=0.001, affine=False)

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

class Conv_Scalar_Binary(nn.Module):
    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Scalar_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op

    def forward(self, x):
        return self.op(self.conv(x), 2.0)

class Conv_Scalar_Binary_Add(nn.Module):
    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Scalar_Binary_Add, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op

    def forward(self, x):
        return torch.add(self.op(self.conv1(x), 2.0), self.op(self.conv2(x), 2.0))

class Conv_Tensor_Binary(nn.Module):
    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Tensor_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op
        input_size = [1, out_channels, 1, 1]
        if dim == 3:
            input_size.append(1)
        self.tensor = torch.randn(input_size)

    def forward(self, x):
        return self.op(self.conv(x), self.tensor)

class Conv_Tensor_Binary2(nn.Module):
    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Tensor_Binary2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op
        input_size = [1, out_channels, 1, 1]
        if dim == 3:
            input_size.append(1)
        self.tensor = torch.randn(input_size, dtype=torch.cfloat)

    def forward(self, x):
        return self.op(self.conv(x), self.tensor)

class Conv_Tensor_Binary_Add(nn.Module):
    def __init__(self, op, dim, in_channels, out_channels, **kwargs):
        super(Conv_Tensor_Binary_Add, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv1 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.conv2 = conv_module[dim](in_channels, out_channels, **kwargs)
        self.op = op
        input_size = [1, out_channels, 1, 1]
        if dim == 3:
            input_size.append(1)
        self.tensor = torch.randn(input_size)

    def forward(self, x):
        return torch.add(self.op(self.conv1(x), self.tensor), self.op(self.conv2(x), self.tensor))

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

class ConvSum_v2(nn.Module):
    def __init__(self, dim, in_channels, out_channels, **kwargs):
        super(ConvSum_v2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)
        self.conv1 = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        a = self.conv(x)
        b = self.conv(x)
        a.add_(b)
        c = self.conv1(x)
        a.add_(c)
        return a

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

class Linear_Scalar_Binary(nn.Module):
    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Scalar_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op

    def forward(self, x):
        return self.op(self.linear(x), 2.0)

class Linear_Tensor_Binary(nn.Module):
    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Tensor_Binary, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op
        self.tensor = torch.randn(out_channels)

    def forward(self, x):
        return self.op(self.linear(x), self.tensor)

class Linear_Tensor_Binary2(nn.Module):
    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Tensor_Binary2, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op
        self.tensor = torch.tensor([2])

    def forward(self, x):
        return self.op(self.linear(x), self.tensor)

class Linear_Tensor_Binary3(nn.Module):
    def __init__(self, op, in_channels, out_channels, **kwargs):
        super(Linear_Tensor_Binary3, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.op = op
        self.tensor = torch.randn(out_channels, dtype=torch.cfloat)

    def forward(self, x):
        return self.op(self.linear(x), self.tensor)


class LinearRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearRelu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.linear(x), inplace=True)

class LinearSigmoidMul(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearSigmoidMul, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)

    def forward(self, x):
        linear_res = self.linear(x)
        return torch.mul(linear_res, F.sigmoid(linear_res))

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

class LinearAddRelu(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, inplace, **kwargs):
        super(LinearAddRelu, self).__init__()
        self.linear = nn.Linear(in_channels, mid_channels, bias=False, **kwargs)
        self.linear1 = nn.Linear(
            mid_channels, out_channels, bias=False, **kwargs)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=False, **kwargs)
        self.inplace = inplace

    def forward(self, x):
        a = self.linear(x)
        a = F.relu(a, inplace=self.inplace)
        a = self.linear1(a)
        b = self.linear2(x)
        return F.relu(a.add_(b), inplace=self.inplace)

class Linear_Reshape_Relu(nn.Module):
    def __init__(self, in_channels, out_channels,dest_shape, **kwargs):
        super(Linear_Reshape_Relu, self).__init__()
        seed = 2018
        torch.manual_seed(seed)
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.dest_shape = dest_shape

    def forward(self, x):
        return F.relu(torch.reshape(self.linear(x),self.dest_shape), inplace=True)

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
        self.pad = (0, 0) * dim
        self.conv = conv_module[dim](in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        y = self.conv(x)
        if y.size(1) != x.size(1):
            z= F.pad(x,
                       self.pad + (0, y.size(1) - x.size(1)), 'constant', 0.)
            y +=z
        else:
            y += x
        return y

class ConvSwishOutplace(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishOutplace, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a1 = self.conv(x)
        b1 = torch.sigmoid(a1)
        c1 = torch.mul(a1, b1)

        return c1

class ConvSwishInplace(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvSwishInplace, self).__init__()
        self.conv = conv_module[dim](in_channels, out_channels, kernel_size, image_size)

    def forward(self, x):
        a = self.conv(x)
        b = torch.sigmoid(a)
        res = a.mul_(b)
        return res

class ConvTranspose(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose, self).__init__()
        self.conv_transpose = convtranspose_module[dim](in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x

class ConvTransposeSigmoidMul(nn.Module):
    def __init__(self, mul, dim, in_channels, out_channels, kernel_size, image_size):
        super(ConvTransposeSigmoidMul, self).__init__()
        self.conv_transpose = convtranspose_module[dim](in_channels, out_channels, kernel_size, image_size)
        self.mul_op = mul

    def forward(self, x):
        a1 = self.conv_transpose(x)
        b1 = torch.sigmoid(a1)
        c1 = self.mul_op(a1, b1)
        return c1    

class ChannelShuffle_with_Static_Shape(nn.Module):
    def __init__(self, batchsize, num_channels, height, width, groups):
        super(ChannelShuffle_with_Static_Shape, self).__init__()
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

class ChannelShuffle_with_Dynamic_Shape(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle_with_Dynamic_Shape, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

class NotChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(NotChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, width, height)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, width, height)
        return x

class MatmulDivOutplaceOutModifiedByOtherOP_v1(nn.Module):
    def __init__(self, div_scalar=False, with_out=True):
        super(MatmulDivOutplaceOutModifiedByOtherOP_v1, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + (y.size()[-1:])
        mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
        mm_out = torch.empty(mm_res_shape, dtype=x.dtype)
        mm_res = torch.matmul(x, y, out=mm_out)
        if self.div_scalar:
            div_res = mm_res.div(2.0)
        else:
            div_res = mm_res.div(torch.ones(mm_res_shape,dtype=x.dtype)+1)
        mm_out.add_(5)
        return div_res

class MatmulDivOutplaceOutModifiedByOtherOP_v2(nn.Module):
    def __init__(self, div_scalar=False, with_out=True):
        super(MatmulDivOutplaceOutModifiedByOtherOP_v2, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + (y.size()[-1:])
        mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
        mm_out = torch.empty(mm_res_shape, dtype=x.dtype)
        mm_res = torch.matmul(x, y, out=mm_out)
        if self.div_scalar:
            div_res = mm_res.div(2.0)
        else:
            div_res = mm_res.div(torch.ones(mm_res_shape,dtype=x.dtype)+1)
        mm_out.add_(5)
        div_out_equal = (mm_out == div_res)
        return div_res + div_out_equal

class MatmulDivOutplace(nn.Module):
    def __init__(self, div_scalar=False, with_out=False):
        super(MatmulDivOutplace, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        mm_res = None
        y = torch.transpose(x, -1, -2).contiguous()
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

class MatmulDivInplace(nn.Module):
    def __init__(self, div_scalar=False, with_out=False):
        super(MatmulDivInplace, self).__init__()
        self.div_scalar = div_scalar
        self.with_out = with_out

    def forward(self, x):
        mm_res = None
        y = torch.transpose(x, -1, -2).contiguous()
        mm_res_shape = x.size()[:-1] + (y.size()[-1:])
        if self.with_out:
            mm_res = torch.randn(mm_res_shape, dtype=x.dtype)
            torch.matmul(x, y, out=mm_res)
        else:
            mm_res = torch.matmul(x, y)
        if self.div_scalar:
            return mm_res.div_(2.0)
        else:
            return mm_res.div_(torch.ones(mm_res_shape,dtype=x.dtype)+1)

class BmmAdd(nn.Module):
    def __init__(self):
        super(BmmAdd, self).__init__()

    def forward(self, input, batch1, batch2):
        bmm_res = torch.bmm(batch1, batch2)
        res = torch.add(bmm_res, input)
        return res

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

class DistilMHAScoresCalculation_v1(nn.Module):
    def __init__(self, dim_per_head, softmax_dim=-1):
        super(DistilMHAScoresCalculation_v1, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, mask):
        mask_shape=[mat1.shape[0],1,1,mat1.shape[3]]
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk.masked_fill_(mask, -float("inf"))
        return self.softmax(qk)

class DistilMHAScoresCalculation_v2(nn.Module):
    def __init__(self, dim_per_head):
        super(DistilMHAScoresCalculation_v2, self).__init__()
        self.dim_per_head = dim_per_head

    def forward(self, mat1, mat2, mask):
        mask_shape=[mat1.shape[0],1,1,mat1.shape[3]]
        mat1 = mat1 / math.sqrt(self.dim_per_head)
        qk = torch.matmul(mat1, mat2.transpose(2, 3))
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk = qk.masked_fill(mask, -float("inf"))
        return nn.functional.softmax(qk, dim=-1)

class Maskedfill__softmax(nn.Module):
    def __init__(self, softmax_dim=-1):
        super(Maskedfill__softmax, self).__init__()
        self.softmax = nn.Softmax(dim=softmax_dim)

    def forward(self, qk, mask):
        mask_shape=[qk.shape[0],1,1,qk.shape[3]]
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk.masked_fill_(mask, -float("inf"))
        return self.softmax(qk)

class Maskedfill_softmax(nn.Module):
    def __init__(self):
        super(Maskedfill_softmax, self).__init__()

    def forward(self, qk, mask):
        mask_shape=[qk.shape[0],1,1,qk.shape[3]]
        mask = (mask == 0).view(mask_shape).expand_as(qk)
        qk = qk.masked_fill(mask, -float("inf"))
        return nn.functional.softmax(qk, dim=-1)

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
        z = torch.add(x,y)
        return self.layernorm(z)


class AddLayerNorm_v1(torch.nn.Module):
    def __init__(self, dim=32):
        super(AddLayerNorm_v1, self).__init__()
        self.layernorm = torch.nn.LayerNorm(dim)
    def forward(self, x, y, z):
        x = x + y + z
        return self.layernorm(x)

class AddLayerNorm_v2(torch.nn.Module):
    def __init__(self, dim=32):
        super(AddLayerNorm_v2, self).__init__()
        self.dim = dim
    def forward(self, x, y, w):
        z = torch.add(x,y)
        return torch.nn.functional.layer_norm(z, [self.dim,], weight=w)

class ConcatBnRelu(torch.nn.Module):
    def __init__(self, dim, cat_dim, in_channels, **kwargs):
        super(ConcatBnRelu, self).__init__()
        self.bn = bn_module[dim](in_channels)
        self.relu = torch.nn.ReLU()
        self.cat_dim = cat_dim
    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim = self.cat_dim)
        x = self.bn(x)
        return self.relu(x)

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

class LinearSwishNaive(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LinearSwishNaive, self).__init__()
        self.linear = nn.Linear(in_feature, out_feature)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        linear_out = self.linear(input)
        sigmoid_out = self.sigmoid(linear_out)
        return torch.mul(linear_out, sigmoid_out)

class Bottleneck_v1(nn.Module):
    def __init__(self):
        super(Bottleneck_v1, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.downsample = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += self.downsample(x)
        return y3.relu_()

class Bottleneck_v2(nn.Module):
    def __init__(self):
        super(Bottleneck_v2, self).__init__()
        self.conv =  nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        x = self.conv(x)
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += x
        return y3.relu_()

class EinsumAdd(nn.Module):
    def __init__(self, equation):
        super(EinsumAdd, self).__init__()
        self.equation = equation
    def forward(self, input1, input2, bias):
        return torch.einsum(self.equation, input1, input2) + bias

class EinsumAddScalar(nn.Module):
    def __init__(self, equation):
        super(EinsumAddScalar, self).__init__()
        self.equation = equation
    def forward(self, input1, input2):
        return torch.einsum(self.equation, input1, input2) + 12.0

class EinsumAddInplace(nn.Module):
    def __init__(self, equation):
        super(EinsumAddInplace, self).__init__()
        self.equation = equation
    def forward(self, input1, input2, bias):
        return torch.einsum(self.equation, input1, input2).add_(bias)

class EinsumAddInplaceV1(nn.Module):
    def __init__(self, equation):
        super(EinsumAddInplaceV1, self).__init__()
        self.equation = equation
    def forward(self, input1, input2, bias):
        return bias.add_(torch.einsum(self.equation, input1, input2))

class AddMulDiv(nn.Module):
    def __init__(self):
        super(AddMulDiv, self).__init__()

    def forward(self, input):
        return torch.div(torch.mul(input, torch.add(input, 3)), 6)

class Tester(TestCase):
    @contextlib.contextmanager
    def _texpr_enable(self, strategy):
        old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(strategy)
        try:
            yield
        finally:
            torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)

    def _test_output(self, base_model, x, kind_in_graph=None, kind_not_in_graph=None, prec=None, levels=['O0','O1'], use_channels_last=[True, False], use_te=[False, True]):
        modelName = base_model.__class__.__name__
        options = itertools.product(levels, use_channels_last, use_te)
        for level, use_channels_last, use_te in options:
            with self._texpr_enable(use_te):
                ipex.enable_onednn_fusion(False)
                model = copy.deepcopy(base_model).eval()
#It will be removed after jit support conv_bn folding
                if level == 'O0':
                    try:
                        model = optimization.fuse(model)
                    except:
                        warnings.warn("Conv BatchNorm folding failed.")
                if x.dim() == 4 and use_channels_last:
                    x = x.to(memory_format=torch.channels_last)
                    model = model.to(memory_format=torch.channels_last)

                if x.dim() == 5 and use_channels_last:
                    x = x.to(memory_format=torch.channels_last_3d)
                    model = model.to(memory_format=torch.channels_last_3d)

                oresult = model(x)

                model = ipex.optimize(model, dtype=torch.float32, level=level)

                with torch.no_grad():
                    result = model(x)
                    traced_model = torch.jit.trace(model, x).eval()
                    traced_model = torch.jit.freeze(traced_model)
                    tresult = traced_model(x)

                self.assertEqual(oresult, result, prec=prec)
                self.assertEqual(result, tresult, prec=prec)

                ipex.enable_onednn_fusion(True)
                with torch.no_grad():
                    trace_fused_model = torch.jit.trace(model, x)
                    trace_fused_model = torch.jit.freeze(trace_fused_model)
                    y = trace_fused_model(x)

#enable fusiong in ipex.
                    fused_tresult = trace_fused_model(x)
#conv relu fusion, conv sum fusion or conv sum relu fusion
                    trace_graph = trace_fused_model.graph_for(x)
                    fused_tresult = trace_fused_model(x)
                self.assertEqual(result, fused_tresult, prec=prec)
#check if the fused node exists in the graph
                if kind_in_graph is not None:
                    self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

#check if certain node does not exist in the graph
                if kind_not_in_graph is not None:
                    self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))

    def _test_onednn_fp32(self, model, input, kind_in_graph=None, kind_not_in_graph=None, prec=5e-3):
        model = model.eval()
        model = ipex.optimize(model, dtype=torch.float32, auto_kernel_selection=True)
        with torch.no_grad():
            res_ref = model(input)
            tr_model = torch.jit.trace(model, (input))
            tr_model = torch.jit.freeze(tr_model)
            tr_model(input)
            trace_graph = tr_model.graph_for(input)
            res_jit = tr_model(input)
            self.assertEqual(res_ref, res_jit)
            
            if kind_in_graph is not None:
                self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))
            
            if kind_not_in_graph is not None:
                self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))


    def _test_output_bf16(self, base_model, x, kind_in_graph=None, kind_not_in_graph=None, prec=None, levels=['O0', 'O1'], use_channels_last=[True, False], use_te=[True, False]):
        modelName = base_model.__class__.__name__
        options = itertools.product(levels, use_channels_last, use_te)
        for level, use_channels_last, use_te in options:
            with self._texpr_enable(use_te):
                ipex.enable_onednn_fusion(True)
                model = copy.deepcopy(base_model).eval()
#It will be removed after jit support conv_bn folding
                if level == 'O0':
                    try:
                        model = optimization.fuse(model)
                    except:
                        warnings.warn("Conv BatchNorm folding failed.")
                if x.dim() == 4 and use_channels_last:
                    x = x.to(memory_format=torch.channels_last)
                    model = model.to(memory_format=torch.channels_last)
                if x.dim() == 5 and use_channels_last:
                    x = x.to(memory_format=torch.channels_last_3d)
                    model = model.to(memory_format=torch.channels_last_3d)

                model = ipex.optimize(model, dtype=torch.bfloat16, level=level)
                x2 = x.clone()
                x3 = x.clone()

                with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16), torch.no_grad():
#bf16, native path
                    result = model(x)
                    trace_fused_model = torch.jit.trace(copy.deepcopy(model), x3)
                    trace_fused_model = torch.jit.freeze(trace_fused_model)
#enable fusion path.
                    fused_tresult = trace_fused_model(x3)
#bf16, jit trace path
                    trace_graph = trace_fused_model.graph_for(x3)
                    fused_tresult = trace_fused_model(x3)

                self.assertEqual(fused_tresult, result, prec=prec)
                self.assertEqual(fused_tresult.dtype, torch.bfloat16)

#check if the fused node exists in the graph
                if kind_in_graph is not None:
                    self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

#check if certain node does not exist in the graph
                if kind_not_in_graph is not None:
                    self.assertTrue(all(n.kind() != kind_not_in_graph for n in trace_graph.nodes()))

    def _test_fusion_unsupported_case(self, m, x, auto_kernel_selection=False, kind_in_graph=None, kind_not_in_graph=None):
        m.eval()
        model = ipex.optimize(m, dtype=torch.float32, auto_kernel_selection=auto_kernel_selection)
        with torch.no_grad():
            result = model(x)
            traced_model = torch.jit.trace(model, x).eval()
            traced_model = torch.jit.freeze(traced_model)
            tresult = traced_model(x)
            trace_graph = traced_model.graph_for(x)                
            
            if kind_in_graph is not None:
                self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

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
#enable fusiong in ipex.
            result1 = trace_model(x)
            result2 = freeze_model(x)
#conv relu fusion, conv sum fusion or conv sum relu fusion
            trace_graph = trace_model.graph_for(x)
            freeze_graph = freeze_model.graph_for(x)

        jit_node = "ipex_prepack::convolution_run"
        pack_node = "ipex_prepack::convolution_prepack"
        imperative_node = "torch_ipex::convolution_forward"
# for freeze model, there will be only convolution_run in the graph
        self.assertTrue(any(n.kind() == jit_node for n in freeze_graph.nodes()))
        self.assertTrue(all(n.kind() != pack_node for n in freeze_graph.nodes()))
# for non-freeze model, since op-ctx dose not have value, cannot re-pack for this path
        self.assertTrue(any(n.kind() == imperative_node for n in trace_graph.nodes()))
        

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
#call mkl path(fp32)
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
#call onednn path(fp32)
        model = ipex.optimize(origin_model, dtype=torch.float32, auto_kernel_selection=True)
        ori_res = model(test_val1)
        model_jit = torch.jit.trace(model,(test_val1))
        graph_ori = str(model_jit.graph_for(test_val1))
        linear_count_ori = check_op_count(graph_ori, ["torch_ipex::ipex_linear"])
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
            linear_count_ori = check_op_count(graph_ori, ["torch_ipex::ipex_linear"])
            self.assertEqual(linear_count_ori, 4)
            model_jit = torch.jit.freeze(model_jit)
            model_jit(test_val1)
            graph_opt = str(model_jit.graph_for(test_val1))
            jit_res = model_jit(test_val1)
            self.assertEqual(ori_res[1], jit_res[1])
            linear_count_ori = check_op_count(graph_opt, ["ipex_prepack::linear_run"])
            self.assertEqual(linear_count_ori, 2)

    def test_add_layernorm(self):
        for dim in [768, 100]:
            with torch.no_grad():
                bs = 56
                seq_len = 384
                a = torch.randn(bs, seq_len, dim)
                b = torch.randn(bs, seq_len, dim)
                w = torch.ones(dim)
                model = AddLayerNorm(dim)
                pre_te_enable_status = torch._C._jit_texpr_fuser_enabled()
                torch._C._jit_set_texpr_fuser_enabled(False)
                jit_model = torch.jit.trace(model,(a, b))
                trace_graph = jit_model.graph_for(a, b)
                jit_res = jit_model(a, b)
                ori_res = model(a, b)
                self.assertEqual(jit_res, ori_res)
                node = "ipex::add_layernorm"
                self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))

                # not contiguous
                a_not_cont = a.clone().detach().unsqueeze(0).to(memory_format=torch.channels_last).squeeze(0)
                b_not_cont = b.clone().detach().unsqueeze(0).to(memory_format=torch.channels_last).squeeze(0)
                ori_res = model(a_not_cont, b_not_cont)
                jit_model = torch.jit.trace(model,(a, b))
                trace_graph = jit_model.graph_for(a, b)
                jit_res = jit_model(a_not_cont, b_not_cont)
                node = "ipex::add_layernorm"
                self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))
                self.assertEqual(jit_res, ori_res)

                a_bf16 = a.to(torch.bfloat16)
                b_bf16 = b.to(torch.bfloat16)
                w_bf16 = w.to(torch.bfloat16)
                model = AddLayerNorm_v2(dim)
                jit_model = torch.jit.trace(model,(a, b, w))
                ori_res = model(a_bf16, b_bf16, w)
                trace_graph = jit_model.graph_for(a_bf16, b_bf16, w_bf16)
                jit_res = jit_model(a_bf16, b_bf16, w_bf16)
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
                torch._C._jit_set_texpr_fuser_enabled(pre_te_enable_status)
                self.assertTrue(any(n.kind() == node for n in trace_graph.nodes()))

    def test_concat_bn_relu(self):
        batch_size = 3
        image_size = 16
        options = itertools.product([2, 3], [[32, 32, 32], [60, 60, 60], [17, 27, 32], [16, 32, 48]], [torch.float32, torch.bfloat16], ['O0', 'O1'], [True, False])
        for dim, channels, dtype, level, use_channels_last in options:
            input_size = [
                [batch_size, channels[0], image_size, image_size],
                [batch_size, channels[1], image_size, image_size],
                [batch_size, channels[2], image_size, image_size]
            ]
            if dim == 3:
                for i in range(3):
                    input_size[i].append(image_size)
            a1 = torch.randn(input_size[0], dtype=dtype)
            a2 = torch.randn(input_size[1], dtype=dtype)
            a3 = torch.randn(input_size[2], dtype=dtype)
            a = [a1, a2, a3]

            in_channels = sum(channels)
            model = ConcatBnRelu(dim, 1, in_channels).eval()

            if use_channels_last:
                suggest_memory_format = torch.channels_last if dim == 2 else torch.channels_last_3d
                for i in range(3):
                    a[i] = a[i].to(memory_format=suggest_memory_format)
                model = model.to(memory_format=suggest_memory_format)

            model = ipex.optimize(model, dtype=dtype, level=level)

            with torch.cpu.amp.autocast(enabled=True if dtype == torch.bfloat16 else False), torch.no_grad():
                result = model(a[0], a[1], a[2])
                trace_model = torch.jit.trace(model, (a[0], a[1], a[2])).eval()
                trace_model = torch.jit.freeze(trace_model)

                tresult = trace_model(a[0], a[1], a[2])
                trace_graph = trace_model.graph_for(a[0], a[1], a[2])

                self.assertEqual(result, tresult)
                self.assertEqual(tresult.dtype, dtype)
                if use_channels_last:
                    self.assertTrue(tresult.is_contiguous(memory_format=suggest_memory_format))
                if use_channels_last and a1.size(1) % 16 == 0 and a2.size(1) % 16 == 0 and a3.size(1) % 16 == 0 :
                    self.assertTrue(any(n.kind() == "ipex::concat_bn_relu" for n in trace_graph.nodes()))
                else:
                    self.assertTrue(all(n.kind() != "ipex::concat_bn_relu" for n in trace_graph.nodes()))

    def test_mha_scores_calculation(self):
        def _check_match_mha(trace_model, mat1, mat2, bias, node = "ipex::mha_scores_calc"):
            graph = trace_model.graph_for((mat1, mat2, bias))
            self.assertTrue(any(n.kind() == node for n in graph.nodes()))

        def _test_pure_bf16(model, trace_model, mat1, mat2, bias, prec=3e-2):
            mat1_bf16 = mat1.to(torch.bfloat16)
            mat2_bf16 = mat2.to(torch.bfloat16)
            bias_bf16 = bias.to(torch.bfloat16)
            res_ref = model(mat1_bf16, mat2_bf16, bias_bf16)
            res_jit = trace_model(mat1_bf16, mat2_bf16, bias_bf16)
            self.assertEqual(res_ref, res_jit, prec=prec)
            _check_match_mha(trace_model, mat1, mat2, bias)

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
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(1, 1, 2, 3)
                mat2 = torch.randn(1, 1, 16, 3)
                bias = torch.randn(1, 1, 2, 16)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(1, 1, 2, 3)
                mat2 = torch.randn(1, 1, 32, 3)
                bias = torch.randn(1, 1, 2, 32)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(1, 1, 2, 3)
                mat2 = torch.randn(1, 1, 33, 3)
                bias = torch.randn(1, 1, 2, 33)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

                mat1 = torch.randn(2, 3, 4, 6)
                mat2 = torch.randn(2, 3, 6, 6)
                bias = torch.randn(2, 3, 4, 6)
                res_ref = mha(mat1, mat2, bias)
                res_jit = mha_jit(mat1, mat2, bias)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

#Test broadcast
                mat1 = torch.randn(2, 3, 4, 10)
                mat2 = torch.randn(2, 3, 16, 10)
                bias = torch.randn(1, 1, 1, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(4, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(3, 1, 1)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(2, 1, 1, 1)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(3, 4, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(2, 1, 1, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)
                bias = torch.randn(2, 1, 4, 16)
                self.assertEqual(mha(mat1, mat2, bias), mha_jit(mat1, mat2, bias))
                _check_match_mha(mha_jit, mat1, mat2, bias)
                _test_pure_bf16(mha, mha_jit, mat1, mat2, bias)

    def test_linear_swish(self):
        mat1 = torch.randn(10000, 5)

        pattern_1 = LinearSwishNaive(5, 1024)

        with torch.no_grad():
            pattern1_jit = torch.jit.trace(pattern_1, (mat1))
            pattern1_jit.eval()
            res_ref = pattern_1(mat1)
            res_jit = pattern1_jit(mat1)

            self.assertEqual(res_ref, res_jit)

            mat2 = torch.randn(10000, 1024)
            pattern_2 = LinearSwishNaive(1024, 1024)
            pattern2_jit = torch.jit.trace(pattern_2, (mat2))
            pattern2_jit.eval()
            res_ref = pattern_2(mat2)
            res_jit = pattern2_jit(mat2)

            self.assertEqual(res_ref, res_jit)

            # non 512 bit align shape
            mat3 = torch.randn(10000, 1019)
            pattern_3 = LinearSwishNaive(1019, 1019)
            pattern3_jit = torch.jit.trace(pattern_3, (mat3))
            pattern3_jit.eval()
            res_ref = pattern_3(mat3)
            res_jit = pattern3_jit(mat3)

            self.assertEqual(res_ref, res_jit)

            def _test_pure_bf16(model, trace_model, mat1, prec=5e-2):
                model = model.to(torch.bfloat16)
                trace_model = trace_model.to(torch.bfloat16)
                mat1_bf16 = mat1.to(torch.bfloat16)
                res_ref = model(mat1_bf16)
                res_jit = trace_model(mat1_bf16)
                self.assertEqual(res_ref, res_jit, prec=prec)

            _test_pure_bf16(pattern_1, pattern1_jit, mat1)
            _test_pure_bf16(pattern_2, pattern2_jit, mat2)
            _test_pure_bf16(pattern_3, pattern3_jit, mat3)

    def test_distil_mha_scores_calculation(self):
        def _check_match_mha(trace_model, mat1, mat2, mask, node = "ipex::distil_mha_scores_calc"):
            graph = trace_model.graph_for((mat1, mat2, mask))
            self.assertTrue(any(n.kind() == node for n in graph.nodes()))

        def _check_match_mha_parts(trace_model, qk, mask, node = "ipex::maskedfill_softmax"):
            graph = trace_model.graph_for((qk, mask))
            self.assertTrue(any(n.kind() == node for n in graph.nodes()))

        def _test_pure_bf16(model, trace_model, mat1, mat2, mask, prec=3e-2):
            mat1_bf16 = mat1.to(torch.bfloat16)
            mat2_bf16 = mat2.to(torch.bfloat16)
            mask_bf16 = mask.to(torch.bfloat16)
            res_ref = model(mat1_bf16, mat2_bf16, mask_bf16)
            res_jit = trace_model(mat1_bf16, mat2_bf16, mask_bf16)
            self.assertEqual(res_ref, res_jit, prec=prec)
            _check_match_mha(trace_model, mat1, mat2, mask)

        def _test_pure_bf16_parts(model, trace_model, qk, mask, prec=3e-2):
            qk_bf16 = qk.to(torch.bfloat16)
            mask_bf16 = mask.to(torch.bfloat16)
            res_ref = model(qk_bf16, mask_bf16)
            res_jit = trace_model(qk_bf16, mask_bf16)
            self.assertEqual(res_ref, res_jit, prec=prec)
            _check_match_mha_parts(trace_model, qk_bf16, mask)
        
        for sequance_length in [128, 100]:
            mat1 = torch.randn(56, 12, sequance_length, sequance_length)
            mat2 = torch.randn(56, 12, sequance_length, sequance_length)
            mask = torch.randn(56, sequance_length)
            qk = torch.matmul(mat1,mat2)
            mask = (mask > 0.5)

            mha_v1 = DistilMHAScoresCalculation_v1(4, -1)
            with torch.no_grad():
                mha_jit = torch.jit.trace(mha_v1, (mat1, mat2, mask))
                mha_jit.eval()

                res_ref = mha_v1(mat1, mat2, mask)
                res_jit = mha_jit(mat1, mat2, mask)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha(mha_jit, mat1, mat2, mask)
                _test_pure_bf16(mha_v1, mha_jit, mat1, mat2, mask)

            mha_v2 = DistilMHAScoresCalculation_v2(4)
            with torch.no_grad():
                mha_jit = torch.jit.trace(mha_v2, (mat1, mat2, mask))
                mha_jit.eval()

                res_ref = mha_v2(mat1, mat2, mask)
                res_jit = mha_jit(mat1, mat2, mask)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha(mha_jit, mat1, mat2, mask)
                _test_pure_bf16(mha_v1, mha_jit, mat1, mat2, mask)

            mha_v3 = Maskedfill__softmax()
            with torch.no_grad():
                mha_jit = torch.jit.trace(mha_v3, (qk, mask))
                mha_jit.eval()

                res_ref = mha_v3(qk, mask)
                res_jit = mha_jit(qk, mask)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha_parts(mha_jit, qk, mask)
                _test_pure_bf16_parts(mha_v3, mha_jit, qk, mask)

            mha_v4 = Maskedfill_softmax()
            with torch.no_grad():
                mha_jit = torch.jit.trace(mha_v4, (qk, mask))
                mha_jit.eval()

                res_ref = mha_v4(qk, mask)
                res_jit = mha_jit(qk, mask)
                self.assertEqual(res_ref, res_jit)
                _check_match_mha_parts(mha_jit, qk, mask)
                _test_pure_bf16_parts(mha_v4, mha_jit, qk, mask)

    def _test_conv_unary_fusion(self, op_list, seed=None):
        batch_size = 8
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16

        if seed is None:
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
        else:
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, seed))     
            torch.manual_seed(seed)

        for dim in [2, 3]:
            for eltwise in op_list:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)

                unary_fusion_op = op_list[eltwise]
                ipex_eltwise_op = unary_fusion_op.ipex_eltwise_op
                bf16_supported = unary_fusion_op.bf16_supported
                prec = unary_fusion_op.prec
                op_input_list = unary_fusion_op.op_input_list

                x = torch.randn(input_size)
                m = ConvEltwise(eltwise, dim, in_channels, out_channels, kernel_size, image_size, **op_input_list)

                self._test_output(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::convolution_%s_run" % ipex_eltwise_op,
                    kind_not_in_graph="ipex_prepack::convolution_%s_prepack" % ipex_eltwise_op)
                if bf16_supported:
                    self._test_output_bf16(
                        m,
                        x,
                        kind_in_graph="ipex_prepack::convolution_%s_run" % ipex_eltwise_op,
                        kind_not_in_graph="ipex_prepack::convolution_%s_prepack" % ipex_eltwise_op,
                        prec=prec)

    def _test_conv_transpose_unary_fusion(self, op_list, seed=None):
        batch_size = 1
        out_channels = 5
        in_channels = 3
        kernel_size = 3
        image_size = 8

        if seed is None:
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
        else:
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, seed))     
            torch.manual_seed(seed)

        for dim in [2, 3]:
            for eltwise in op_list:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)

                unary_fusion_op = op_list[eltwise]
                ipex_eltwise_op = unary_fusion_op.ipex_eltwise_op
                bf16_supported = unary_fusion_op.bf16_supported
                prec = unary_fusion_op.prec
                op_input_list = unary_fusion_op.op_input_list

                x = torch.randn(input_size)
                m = ConvTransposeEltwise(eltwise, dim, in_channels, out_channels, kernel_size, image_size, **op_input_list)

                self._test_output(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                    kind_not_in_graph="ipex_prepack::conv_transpose_prepack")
                if bf16_supported:
                    self._test_output_bf16(
                        m,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                        prec=prec)

    def test_conv_unary_fusion(self):
        self._test_conv_unary_fusion(unary_PyTorch_op_to_IPEX_op_map)
        self._test_conv_unary_fusion(PyTorch_op_to_IPEX_op_fixed_seed_map, 1654064339261196288)

    def test_conv_non_unary_fusion(self):        
        self._test_conv_unary_fusion(non_unary_PyTorch_op_to_IPEX_op_map)

    def test_conv_fusion_unsupported_case(self):
        dim = 2
        batch_size = 1
        in_channels = 3
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16      
        for eltwise in unsupported_PyTorch_op_to_IPEX_op_map:
            input_size = [batch_size, in_channels, image_size, image_size]

            unary_fusion_op = unsupported_PyTorch_op_to_IPEX_op_map[eltwise]
            ipex_eltwise_op = unary_fusion_op.ipex_eltwise_op
            bf16_supported = unary_fusion_op.bf16_supported
            prec = unary_fusion_op.prec
            op_input_list = unary_fusion_op.op_input_list

            x = torch.randn(input_size)
            m = ConvEltwise(eltwise, dim, in_channels, out_channels, kernel_size, image_size, **op_input_list)
            
            self._test_fusion_unsupported_case(
                m,
                x,
                kind_not_in_graph = 'ipex_prepack::convolution_%s_run' % ipex_eltwise_op)

    def _test_conv_transpose_sum(self, module, alpha, supported, test_inplace=True):
        batch_size = 1
        out_channels = 3
        in_channels = 3
        kernel_size = 3
        image_size = 8

        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        prec = 0.02
        kwargs = {"alpha": alpha}
        add_funcs = [lambda a, b, kwargs: torch.add(a, b, **kwargs)]
        if test_inplace:
            add_funcs.append(lambda a, b, kwargs: a.add_(b, **kwargs))
        
        for dim in [2, 3]:
            for add_func in add_funcs:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                ipex_eltwise_op = "add"
                x = torch.randn(input_size)
                m = module(dim, add_func, in_channels, out_channels, kernel_size, image_size, **kwargs)
                if supported:
                    self._test_output(
                        m,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack")
                    self._test_output_bf16(
                        m,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                        prec=prec)
                else:
                    self._test_output(
                        m,
                        x,
                        kind_not_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op)
                    self._test_output_bf16(
                        m,
                        x,
                        kind_not_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                        prec=prec)

    def test_conv_transpose_sum_accumu_on_right(self):
        self._test_conv_transpose_sum(
            ConvTransposeSumAccumuOnRight, 
            alpha=1,
            supported=True)
        self._test_conv_transpose_sum(
            ConvTransposeSumAccumuOnRight,
            alpha=2,
            supported=True)

    def test_conv_transpose_sum_accumu_on_left(self):
        self._test_conv_transpose_sum(
            ConvTransposeSumAccumuOnLeft,
            alpha=1,
            supported=True)
        self._test_conv_transpose_sum(
            ConvTransposeSumAccumuOnLeft,
            alpha=2,
            supported=False)
        self._test_conv_transpose_sum(
            ConvTransposeSumAccumuOnLeft,
            alpha=2.,
            supported=False)

    def test_conv_transpose_sum_broadcast_unsupported(self):
        self._test_conv_transpose_sum(
            ConvTransposeSumBroadcast,
            alpha=1,
            supported=False,
            test_inplace=False) # in-place add does not support shape broadcast

    def test_conv_transpose_sum_relu(self):
        batch_size = 1
        out_channels = 3
        mid_channels = 2
        in_channels = 3
        kernel_size = 3
        image_size = 8        
        for inplace in [True, False]:
            for dim in [2, 3]:
                m = ConvTransposeAddRelu(dim, in_channels, mid_channels, out_channels, kernel_size, inplace)

                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                x = torch.randn(input_size)
                
                self._test_output(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose_add_relu_run",
                    kind_not_in_graph="ipex_prepack::conv_transpose_add_run")                
                self._test_output_bf16(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose_add_relu_run",
                    kind_not_in_graph="ipex_prepack::conv_transpose_add_run",
                    prec=5e-2)        

    def test_conv_fusion(self):
        batch_size = 8
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            '''
            self._test_output(
                ConvSwishOutplace(in_channels, out_channels, kernel_size, image_size),
                torch.randn(batch_size, in_channels, image_size, image_size),
                kind_in_graph="ipex_prepack::convolution_swish_run")
            '''
            self._test_output_bf16(
                ConvSwishOutplace(dim, in_channels, out_channels, kernel_size, image_size),
                x,
                kind_in_graph="ipex_prepack::convolution_swish_run",
                kind_not_in_graph="ipex_prepack::convolution_swish_prepack",
                prec=0.02)
            self._test_output(
                ConvSwishInplace(dim, in_channels, out_channels, kernel_size, image_size),
                x,
                kind_in_graph="ipex_prepack::convolution_swish_run",
                kind_not_in_graph="ipex_prepack::convolution_swish_prepack")
            self._test_output_bf16(
                ConvSwishInplace(dim, in_channels, out_channels, kernel_size, image_size),
                x,
                kind_in_graph="ipex_prepack::convolution_swish_run",
                kind_not_in_graph="ipex_prepack::convolution_swish_prepack",
                prec=0.02)

    def test_output_conv_bn(self):
        batch_size = 8
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                ConvBatchNorm_Fixed(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex::batch_norm",
                levels=['O1'])
            self._test_output_bf16(
                ConvBatchNorm_Fixed(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex::batch_norm",
                prec=0.02,
                levels=['O1'])
            self._test_output(
                ConvBatchNorm_Fixed2(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex::batch_norm",
                prec=0.02,
                levels=['O0'])

    def test_output_frozen_conv_bn(self):
        batch_size = 8
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16
        options = itertools.product([torch.float32, torch.bfloat16], [True, False], [ConvBatchNorm_Fixed, ConvBatchNorm_Fixed3])
        for dtype, use_channels_last, model in options:
            input_size = [batch_size, in_channels, image_size, image_size]
            model = model(2, in_channels, out_channels, kernel_size=kernel_size, stride=1).eval()
            x = torch.randn(input_size, dtype=dtype)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)
                model = model.to(memory_format=torch.channels_last)
            
            model = ipex.optimize(model, dtype=dtype, conv_bn_folding=False)

            with torch.cpu.amp.autocast(enabled=True, dtype=dtype), torch.no_grad():
                result = model(x)
                trace_model = torch.jit.trace(model, x).eval()
                freeze_model = torch.jit.freeze(trace_model)

                tresult = trace_model(x)
                fused_tresult = freeze_model(x)

                trace_graph = trace_model.graph_for(x)
                freeze_graph = freeze_model.graph_for(x)

                self.assertEqual(result, tresult)
                self.assertEqual(result, fused_tresult)
                self.assertEqual(fused_tresult.dtype, dtype)
                self.assertTrue(any(n.kind() == "ipex::batch_norm" for n in trace_graph.nodes()))
                self.assertTrue(all(n.kind() != "ipex::batch_norm" for n in freeze_graph.nodes()))

    def test_output_bn_conv(self):
        batch_size = 8
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                BatchNormConv_Fixed(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex::batch_norm",
                kind_not_in_graph=None)

    def test_output_bn_conv_bn(self):
        batch_size = 8
        out_channels = 16
        in_channels = 3
        kernel_size = 3
        image_size = 16
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                BatchNorm_Conv_BatchNorm(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex::batch_norm",
                kind_not_in_graph=None)

    def test_output_conv_reshape_bn(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            dst_shape = [16, 16, 62, 62]
            if dim == 3:
                dst_shape.append(62)
            self._test_output(
                ConvReshapeBatchNorm(dim, in_channels, out_channels, dst_shape, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex::batch_norm",
                kind_not_in_graph=None)

    def test_output_conv_conv_concate(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                Conv_Conv_Concat(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex_prepack::convolution_prepack")

    def test_output_conv_relu_add(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                Conv_Relu_Add(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_relu_run",
                kind_not_in_graph="ipex_prepack::convolution_relu_prepack")

    def test_output_conv_scalar_binary(self):
        batch_size = 2
        out_channels = 12
        in_channels = 3
        kernel_size = 3
        image_size = 24
        for dim in [2, 3]:
            for bias in [True, False]:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                x = torch.randn(input_size)
                self._test_output(
                    Conv_Scalar_Binary(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::add")

                self._test_output(
                    Conv_Scalar_Binary(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::sub")

                self._test_output(
                    Conv_Scalar_Binary(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::mul")

                self._test_output(
                    Conv_Scalar_Binary(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::div")

                self._test_output_bf16(
                    Conv_Scalar_Binary(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::add",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Scalar_Binary(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::sub",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Scalar_Binary(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::mul",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Scalar_Binary(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::div",
                    prec=0.1)

    def test_output_conv_scalar_binary_add(self):
        batch_size = 2
        out_channels = 12
        in_channels = 3
        kernel_size = 3
        image_size = 24
        for dim in [2, 3]:
            for bias in [True, False]:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                x = torch.randn(input_size)
                self._test_output(
                    Conv_Scalar_Binary_Add(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::add")

                self._test_output(
                    Conv_Scalar_Binary_Add(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::sub")

                self._test_output(
                    Conv_Scalar_Binary_Add(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::mul")

                self._test_output(
                    Conv_Scalar_Binary_Add(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::div")

                self._test_output_bf16(
                    Conv_Scalar_Binary_Add(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::add",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Scalar_Binary_Add(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::sub",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Scalar_Binary_Add(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::mul",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Scalar_Binary_Add(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::div",
                    prec=0.1)

    def test_output_conv_tensor_binary(self):
        batch_size = 2
        out_channels = 12
        in_channels = 3
        kernel_size = 3
        image_size = 24
        for dim in [2, 3]:
            for bias in [True, False]:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                x = torch.randn(input_size)
                self._test_output(
                    Conv_Tensor_Binary2(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="aten::add")

                self._test_output(
                    Conv_Tensor_Binary(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::add")

                self._test_output(
                    Conv_Tensor_Binary(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::sub")

                self._test_output(
                    Conv_Tensor_Binary(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::mul")

                self._test_output(
                    Conv_Tensor_Binary(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::div",
                    prec=2e-5)

                self._test_output_bf16(
                    Conv_Tensor_Binary(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::add",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Tensor_Binary(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::sub",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Tensor_Binary(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::mul",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Tensor_Binary(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_run",
                    kind_not_in_graph="aten::div",
                    prec=0.5)

    def test_output_conv_tensor_binary_add(self):
        batch_size = 2
        out_channels = 12
        in_channels = 3
        kernel_size = 3
        image_size = 24
        for dim in [2, 3]:
            for bias in [True, False]:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                x = torch.randn(input_size)
                self._test_output(
                    Conv_Tensor_Binary_Add(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::add")

                self._test_output(
                    Conv_Tensor_Binary_Add(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::sub")

                self._test_output(
                    Conv_Tensor_Binary_Add(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::mul")

                self._test_output(
                    Conv_Tensor_Binary_Add(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::div",
                    prec=2e-5)

                self._test_output_bf16(
                    Conv_Tensor_Binary_Add(torch.add, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::add",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Tensor_Binary_Add(torch.sub, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::sub",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Tensor_Binary_Add(torch.mul, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::mul",
                    prec=0.1)

                self._test_output_bf16(
                    Conv_Tensor_Binary_Add(torch.div, dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias),
                    x,
                    kind_in_graph="ipex_prepack::convolution_add_run",
                    kind_not_in_graph="aten::div",
                    prec=0.5)

    def test_output_conv_bn_relu(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                Conv_Bn_Relu(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_relu_run",
                kind_not_in_graph="ipex_prepack::convolution_relu_prepack")

    def test_output_conv_reshape_relu(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            dst_shape = [16, 16, 62, 62]
            if dim == 3:
                dst_shape.append(62)
        self._test_output(
            ConvReshapeRelu(dim, in_channels, out_channels, dst_shape, kernel_size=kernel_size, stride=1),
            x,
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_relu_run")

    def test_output_conv_reshape_sum(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            dst_shape = [16, 16, 62, 62]
            if dim == 3:
                dst_shape.append(62)

        self._test_output(
            ConvReshapeSum(dim, in_channels, out_channels, dst_shape, kernel_size=kernel_size, stride=1),
            x,
            kind_in_graph="ipex_prepack::convolution_run",
            kind_not_in_graph="ipex_prepack::convolution_add_run")

    def test_output_conv_sum(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [1, 2, 3]:
            if dim == 1:
                input_size = [batch_size, in_channels, image_size]
            else:
                input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)
            self._test_output(
                ConvSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_add_run",
                kind_not_in_graph="ipex_prepack::convolution_add_prepack")
            self._test_output_bf16(
                ConvSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_add_run",
                kind_not_in_graph="ipex_prepack::convolution_add_prepack",
                prec=0.1)
            self._test_output(
                ConvSum_v2(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_add_run",
                kind_not_in_graph="ipex_prepack::convolution_add_prepack")
            self._test_output_bf16(
                ConvSum_v2(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_add_run",
                kind_not_in_graph="ipex_prepack::convolution_add_prepack",
                prec=0.1)

#add outputs' have different data format
            m = ConvSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1).eval()
            if dim == 2:
                m.conv = m.conv.to(memory_format=torch.torch.channels_last)
            else:
                m.conv = m.conv.to(memory_format=torch.torch.channels_last_3d)
            self._test_output(
                m,
                x,
                kind_in_graph="ipex_prepack::convolution_add_run",
                kind_not_in_graph="ipex_prepack::convolution_add_prepack",
                use_channels_last=[False])
            m = ConvSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1).eval()
            if dim == 2:
                m.conv = m.conv.to(memory_format=torch.channels_last)
            else:
                m.conv = m.conv.to(memory_format=torch.channels_last_3d)
            self._test_output_bf16(
                m,
                x,
                kind_in_graph="ipex_prepack::convolution_add_run",
                kind_not_in_graph="ipex_prepack::convolution_add_prepack",
                prec=0.1,
                use_channels_last=[False])

    def test_conv_sum_dynamic_shape(self):
        m = ConvSum(2, 3, 16, kernel_size=3, stride=1).eval()
        x1 = torch.randn(1, 3, 56, 56)
        x2 = torch.randn(2, 3, 56, 56)
        with torch.no_grad():
            traced = torch.jit.trace(m, x1)
            traced = torch.jit.freeze(traced)
            # apply fusion
            y = m(x1)
            y = m(x1)
            traced_y = traced(x2)
            eager_y = m(x2)
            self.assertEqual(eager_y, traced_y)

    def test_output_conv_scalar_sum(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)

            self._test_output(
                ConvScalarSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex_prepack::convolution_add_run")
            self._test_output_bf16(
                ConvScalarSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex_prepack::convolution_add_run",
                prec=0.1)

    def test_output_conv_broadcast_sum(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)

            self._test_output(
                ConvBroadcastSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex_prepack::convolution_add_run")
            self._test_output_bf16(
                ConvBroadcastSum(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_run",
                kind_not_in_graph="ipex_prepack::convolution_add_run",
                prec=0.1)

    def test_output_cascaded_conv_bn_sum_relu(self):
        batch_size = 8
        mid_channels = 64
        out_channels = 32
        in_channels = 3
        kernel_size = 3
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)

            self._test_output(
                CascadedConvBnSumRelu(dim, in_channels, mid_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_add_relu_run",
                kind_not_in_graph="ipex::batch_norm")
            self._test_output_bf16(
                CascadedConvBnSumRelu(dim, in_channels, mid_channels, out_channels, kernel_size=kernel_size, stride=1),
                x,
                kind_in_graph="ipex_prepack::convolution_add_relu_run",
                kind_not_in_graph="ipex::batch_norm",
                prec=0.02)

    def test_bottleneck_fusion(self):
        x1 = torch.randn(1, 64, 56, 56)
        self._test_output(
            Bottleneck_v1(),
            x1,
            kind_in_graph="ipex_prepack::convolution_bottleneck_run",
            use_channels_last=[True],
            levels=['O1'])
        self._test_output_bf16(
            Bottleneck_v1(),
            x1,
            kind_in_graph="ipex_prepack::convolution_bottleneck_run",
            prec=0.03,
            use_channels_last=[True],
            levels=['O1'])
        self._test_output(
            Bottleneck_v2(),
            x1,
            kind_in_graph="ipex_prepack::convolution_bottleneck_run",
            use_channels_last=[True],
            levels=['O1'])
        self._test_output_bf16(
            Bottleneck_v2(),
            x1,
            kind_in_graph="ipex_prepack::convolution_bottleneck_run",
            prec=0.03,
            use_channels_last=[True],
            levels=['O1'])
        # dynamic shape
        models = [Bottleneck_v1().eval(), Bottleneck_v2().eval()]
        x2 = torch.randn(2, 64, 56, 56)
        with torch.no_grad():
            for m in models:
                traced = torch.jit.trace(m, x1)
                traced = torch.jit.freeze(traced)
                # apply fusion
                y = m(x1)
                y = m(x1)
                traced_y = traced(x2)
                eager_y = m(x2)
                self.assertEqual(eager_y, traced_y)

    def test_jit_conv_sum_in_diff_block(self):
        batch_size = 8
        out_channels = 32
        in_channels = 3
        kernel_size = 1
        image_size = 64
        for dim in [2, 3]:
            input_size = [batch_size, in_channels, image_size, image_size]
            if dim == 3:
                input_size.append(image_size)
            x = torch.randn(input_size)

            self._test_output(
                ConvSumInDiffBlock(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0),
                x,
                kind_not_in_graph="ipex_prepack::convolution_add_run")
            self._test_output_bf16(
                ConvSumInDiffBlock(dim, in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0),
                x,
                kind_not_in_graph="ipex_prepack::convolution_add_run")

    def test_output_conv_transpose(self):
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

        def _deconv_with_output_padding():
            params_dict = {
                "input_height": 8,
                "input_width": 8,
                "input_depth": 8,
                "input_channel_per_group": 10,
                "output_channel_per_group": 10,
                "kernel_size": 3,
                "bias": False,
                "stride": 2,
                "padding": 1,
                "output_padding": 2,
                "groups": 1,
                "dilation": 3,
            }

            params_list = []

            for key, value in params_dict.items():
                params_list.append(value)
            return params_list

        params_list = _deconv_params_list()

        for input_width, input_height, input_depth, input_channel_per_group, output_channel_per_group, kernel_size, bias, stride, padding, output_padding, groups, dilation in list(itertools.product(*params_list)) + [_deconv_with_output_padding()]:
            if (output_padding < stride or output_padding < dilation) \
                    and ((input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0) \
                    and ((input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0) \
                    and ((input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1 > 0):

                ic = input_channel_per_group * groups
                oc = output_channel_per_group * groups

                for dim in [2, 3]:
                    if dim == 2:
                        x = torch.randn(2, ic, input_height, input_width)
                    else:
                        x = torch.randn(2, ic, input_depth, input_height, input_width)
                    model = ConvTranspose(dim, ic, oc, kernel_size, stride, padding, output_padding, groups, bias, dilation)

                    self._test_output(
                        model,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_run",
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                        levels=["O0"])
                    self._test_output_bf16(
                        model,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_run",
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                        levels=["O0"],
                        prec=0.02)
                    self._test_output(
                        model,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_run",
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                        levels=["O1"])
                    self._test_output_bf16(
                        model,
                        x,
                        kind_in_graph="ipex_prepack::conv_transpose_run",
                        kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                        levels=["O1"],
                        prec=0.02)

    def test_conv_transpose_unary_fusion(self):
        self._test_conv_transpose_unary_fusion(unary_PyTorch_op_to_IPEX_op_map)
        self._test_conv_transpose_unary_fusion(PyTorch_op_to_IPEX_op_fixed_seed_map, 1654583254233936896)

    def test_conv_transpose_non_unary_fusion(self):        
        self._test_conv_transpose_unary_fusion(non_unary_PyTorch_op_to_IPEX_op_map)

    def test_conv_transpose_fusion_unsupported_case(self):
        dim = 2
        batch_size = 1
        in_channels = 3
        out_channels = 5
        in_channels = 3
        kernel_size = 3
        image_size = 8
        
        for eltwise in unsupported_PyTorch_op_to_IPEX_op_map:
            input_size = [batch_size, in_channels, image_size, image_size]

            unary_fusion_op = unsupported_PyTorch_op_to_IPEX_op_map[eltwise]
            ipex_eltwise_op = unary_fusion_op.ipex_eltwise_op
            bf16_supported = unary_fusion_op.bf16_supported
            prec = unary_fusion_op.prec
            op_input_list = unary_fusion_op.op_input_list

            x = torch.randn(input_size)
            m = ConvTransposeEltwise(eltwise, dim, in_channels, out_channels, kernel_size, image_size, **op_input_list)

            self._test_fusion_unsupported_case(
                m,
                x,
                kind_not_in_graph = 'ipex_prepack::conv_transpose_%s_run' % ipex_eltwise_op)

    def test_conv_transpose_sigmoid_mul(self):
        batch_size = 1
        out_channels = 5
        in_channels = 3
        kernel_size = 3
        image_size = 8

        rand_seed = int(get_rand_seed())
        print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
        torch.manual_seed(rand_seed)
        prec = 0.02
        for dim in [2, 3]:
            for eltwise in [torch.mul, lambda a, b: a.mul_(b)]:
                input_size = [batch_size, in_channels, image_size, image_size]
                if dim == 3:
                    input_size.append(image_size)
                ipex_eltwise_op = "swish"
                x = torch.randn(input_size)
                m = ConvTransposeSigmoidMul(eltwise, dim, in_channels, out_channels, kernel_size, image_size)

                self._test_output(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                    kind_not_in_graph="ipex_prepack::conv_transpose_prepack")
                self._test_output_bf16(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::conv_transpose_%s_run" % ipex_eltwise_op,
                    kind_not_in_graph="ipex_prepack::conv_transpose_prepack",
                    prec=prec)        

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
# for auto_select_kernel is True and level is O1, we will use ipex linear
                    self.assertTrue(any(n.kind() == 'ipex_prepack::linear_relu_run' for n in trace_graph.nodes()))
                else:
# auto_select_kernel is false, we will use mkl linear
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

#for bfloat16 path, we will use ipex linear for 'O0' and 'O1'
                self.assertTrue(any(n.kind() == 'ipex_prepack::linear_relu_run' for n in trace_graph.nodes()))

    def test_output_linear_scalar_binary(self):
        for bias in [True, False]:
            self._test_output(
                Linear_Scalar_Binary(torch.add, 3, 1, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::add")

            self._test_output(
                Linear_Scalar_Binary(torch.add, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::add")

            self._test_output(
                Linear_Scalar_Binary(torch.sub, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::sub")

            self._test_output(
                Linear_Scalar_Binary(torch.mul, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::mul")

            self._test_output(
                Linear_Scalar_Binary(torch.div, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::div")

            self._test_output_bf16(
                Linear_Scalar_Binary(torch.add, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::add",
                prec=0.1)

            self._test_output_bf16(
                Linear_Scalar_Binary(torch.sub, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::sub",
                prec=0.1)

            self._test_output_bf16(
                Linear_Scalar_Binary(torch.mul, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::mul",
                prec=0.1)

            self._test_output_bf16(
                Linear_Scalar_Binary(torch.div, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::div",
                prec=0.1)

    def test_output_linear_tensor_binary(self):
        for bias in [True, False]:
            self._test_output(
                Linear_Tensor_Binary2(torch.add, 3, 2, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::add")

            self._test_output(
                Linear_Tensor_Binary3(torch.add, 3, 2, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::add")

            self._test_output(
                Linear_Tensor_Binary(torch.add, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::add")

            self._test_output(
                Linear_Tensor_Binary(torch.sub, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::sub")

            self._test_output(
                Linear_Tensor_Binary(torch.mul, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::mul")

            self._test_output(
                Linear_Tensor_Binary(torch.div, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="aten::linear",
                kind_not_in_graph="aten::div")

            self._test_output_bf16(
                Linear_Tensor_Binary(torch.add, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::add",
                prec=0.1)

            self._test_output_bf16(
                Linear_Tensor_Binary(torch.sub, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::sub",
                prec=0.1)

            self._test_output_bf16(
                Linear_Tensor_Binary(torch.mul, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::mul",
                prec=0.1)

            self._test_output_bf16(
                Linear_Tensor_Binary(torch.div, 3, 32, bias=bias),
                torch.randn(52, 3),
                kind_in_graph="ipex_prepack::linear_run",
                kind_not_in_graph="aten::div",
                prec=0.2)

    def _test_linear_unary_fusion(self, op_list, seed=None):
        batch_size = 3
        out_channels = 32
        in_channels = 3

        if seed is None:
            rand_seed = int(get_rand_seed())
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, rand_seed))
            torch.manual_seed(rand_seed)
        else:
            print("{} rand sed: {}".format(sys._getframe().f_code.co_name, seed))     
            torch.manual_seed(seed)         

        for bias in [True, False]:
            for eltwise in op_list:
                input_size = [batch_size, in_channels]
                
                unary_fusion_op = op_list[eltwise]
                ipex_eltwise_op = unary_fusion_op.ipex_eltwise_op
                bf16_supported = unary_fusion_op.bf16_supported
                prec = unary_fusion_op.prec
                op_input_list = unary_fusion_op.op_input_list
                
                x = torch.randn(input_size)
                m = LinearEltwise(eltwise, in_channels, out_channels, bias, **op_input_list)
                
                self._test_output(
                    m,
                    x,
                    kind_in_graph="aten::linear")
                self._test_onednn_fp32(
                    m,
                    x,
                    kind_in_graph="ipex_prepack::linear_%s_run" % ipex_eltwise_op,
                    kind_not_in_graph="ipex_prepack::linear_prepack")                
                if bf16_supported:
                    self._test_output_bf16(
                        m,
                        x,
                        kind_in_graph="ipex_prepack::linear_%s_run" % ipex_eltwise_op,
                        kind_not_in_graph="ipex_prepack::linear_prepack",
                        prec=prec)

    def test_linear_unary_fusion(self):
        self._test_linear_unary_fusion(unary_PyTorch_op_to_IPEX_op_map)
        self._test_linear_unary_fusion(PyTorch_op_to_IPEX_op_fixed_seed_map, 1654065112450588160)

    def test_linear_non_unary_fusion(self):
        self._test_linear_unary_fusion(non_unary_PyTorch_op_to_IPEX_op_map)

    def test_linear_fusion_unsupported_case(self):
        batch_size = 3
        out_channels = 32
        in_channels = 3
        bias = False       

        for eltwise in unsupported_PyTorch_op_to_IPEX_op_map:
            input_size = [batch_size, in_channels]
            
            unary_fusion_op = unsupported_PyTorch_op_to_IPEX_op_map[eltwise]
            ipex_eltwise_op = unary_fusion_op.ipex_eltwise_op
            bf16_supported = unary_fusion_op.bf16_supported
            prec = unary_fusion_op.prec
            op_input_list = unary_fusion_op.op_input_list
            
            x = torch.randn(input_size)
            m = LinearEltwise(eltwise, in_channels, out_channels, bias, **op_input_list)

            self._test_fusion_unsupported_case(
                m,
                x,
                auto_kernel_selection=True,
                kind_not_in_graph='ipex_prepack::linear_%s_run' % ipex_eltwise_op)

    def test_output_linear_add(self):
        self._test_output(
            LinearAdd(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="aten::linear")

    def test_output_linear_add_relu(self):
        for inplace in [True, False]:
            m = LinearAddRelu(3, 5, 8, inplace)
            x = torch.randn(2, 3)
            self._test_output(
                m,
                x,
                kind_in_graph="aten::linear")        
            self._test_onednn_fp32(
                m,
                x,
                kind_in_graph="ipex_prepack::linear_add_relu_run",
                kind_not_in_graph="ipex_prepack::linear_add_run")                
            self._test_output_bf16(
                m,
                x,
                kind_in_graph="ipex_prepack::linear_add_relu_run",
                kind_not_in_graph="ipex_prepack::linear_add_run",
                prec=5e-2)

    def test_output_linear_reshape_relu(self):
        self._test_output(
            Linear_Reshape_Relu(3, 32,(64,16),bias=True),
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

    def test_output_linear_swish(self):
        self._test_onednn_fp32(
            LinearSigmoidMul(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_swish_run")

        self._test_output_bf16(
            LinearSigmoidMul(3, 32, bias=True),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_swish_run",
            prec=5e-3)
        self._test_onednn_fp32(
            LinearSigmoidMul(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_swish_run")
        self._test_output_bf16(
            LinearSigmoidMul(3, 32, bias=False),
            torch.rand(32, 3),
            kind_in_graph="ipex_prepack::linear_swish_run",
            prec=5e-3)

    def test_channel_shuffle(self):
        self._test_output(
            ChannelShuffle_with_Static_Shape(10, 16, 50, 50, 4),
            torch.rand(10, 16, 50, 50),
            kind_in_graph="ipex::shuffle_2d")
        self._test_output(
            ChannelShuffle_with_Dynamic_Shape(4),
            torch.rand(10, 16, 50, 50),
            kind_in_graph="ipex::shuffle_2d")
        self._test_output(
            NotChannelShuffle(4),
            torch.rand(10, 16, 50, 60),
            kind_not_in_graph="ipex::shuffle_2d")


    def test_jit_function(self):
#test hool trace and script can works for function
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

    def test_matmul_div(self):
        inputs = [torch.randn(10, 3, 4), torch.randn(3, 4)]
        for x in inputs:
            self._test_output(
                MatmulDivOutplace(div_scalar=True, with_out=True),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output(
                MatmulDivOutplace(div_scalar=True, with_out=False),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output(
                MatmulDivOutplace(div_scalar=False, with_out=False),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output(
                MatmulDivOutplace(div_scalar=False, with_out=True),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output_bf16(
                MatmulDivOutplace(div_scalar=True, with_out=True),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-2)
            self._test_output_bf16(
                MatmulDivOutplace(div_scalar=True, with_out=False),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-2)
            self._test_output_bf16(
                MatmulDivOutplace(div_scalar=False, with_out=True),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-3)
            self._test_output_bf16(
                MatmulDivOutplace(div_scalar=False, with_out=False),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-3)
            self._test_output(
                MatmulDivInplace(div_scalar=True, with_out=True),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output(
                MatmulDivInplace(div_scalar=True, with_out=False),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output(
                MatmulDivInplace(div_scalar=False, with_out=False),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output(
                MatmulDivInplace(div_scalar=False, with_out=True),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            self._test_output_bf16(
                MatmulDivInplace(div_scalar=True, with_out=True),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-2)
            self._test_output_bf16(
                MatmulDivInplace(div_scalar=True, with_out=False),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-2)
            self._test_output_bf16(
                MatmulDivInplace(div_scalar=False, with_out=True),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-3)
            self._test_output_bf16(
                MatmulDivInplace(div_scalar=False, with_out=False),
                x.to(torch.bfloat16),
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None,
                prec=5e-3)
            # When the div is outplace and out parameter be modified with an inplace op not in this pattern,
            # but we didn't observe it's value.
            self._test_output(
                MatmulDivOutplaceOutModifiedByOtherOP_v1(div_scalar=True),
                x,
                kind_in_graph="ipex::matmul_div",
                kind_not_in_graph=None)
            # When the div is outplace and out parameter be modified with an inplace op not in this pattern,
            # and we observe it's value by some other op("==" -> aten::equl). In this case, jit.trace will treat 
            # out parameter that will modified by other ops as output of matmul, thus will not be matched by
            # our pattern, and we can't observe our fused op's side effect after we modified out param by an inplace op.
            self._test_output(
                MatmulDivOutplaceOutModifiedByOtherOP_v2(div_scalar=False),
                x,
                kind_in_graph=None,
                kind_not_in_graph="ipex::matmul_div")

    def test_bmm_add(self):
        M = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        mod = BmmAdd()
        traced_mod = torch.jit.trace(mod, (M, batch1, batch2))
        fused_mod = traced_mod.graph_for(M, batch1, batch2)
        out = traced_mod(M, batch1, batch2)
        expected = torch.baddbmm(M, batch1, batch2)
        self.assertTrue(torch.allclose(out, expected))

    def test_einsum_add(self):
        def _test_fp32(model_test, input1, input2, bias=None, kind_in_graph='ipex::einsum_binary', prec=1e-3):
            model = copy.deepcopy(model_test)
            model = model.eval()
            model = ipex.optimize(model, dtype=torch.float32)
            with torch.no_grad():
                res_ref = model(input1, input2, bias)
                tr_model = torch.jit.trace(model, (input1, input2, bias))
                tr_model = torch.jit.freeze(tr_model)
                tr_model(input1, input2, bias)
                tr_model(input1, input2, bias)
                trace_graph = tr_model.graph_for(input1, input2, bias)
                res_jit = tr_model(input1, input2, bias,)
                self.assertEqual(res_ref, res_jit, prec)
                self.assertTrue(any(n.kind() == kind_in_graph for n in trace_graph.nodes()))

        bias = torch.randn(2, 3, 2304)
        input1 = torch.randn(2, 3, 768)
        input2 = torch.randn(768, 2304)
        model_v1 = EinsumAdd('bsh,ho->bso')
        _test_fp32(model_v1, input1, input2, bias)
       
        bias = torch.randn(1, 1, 1, 4)
        input1 = torch.randn(12, 1, 4, 16)
        input2 = torch.randn(12, 4, 4, 16)
        model_v1 = EinsumAdd('bqhc,bkhc->bhqk')
        _test_fp32(model_v1, input1, input2, bias)

        bias = torch.randn(2304)
        input1 = torch.randn(4, 3, 768)
        input2 = torch.randn(768, 2304)
        model_v1 = EinsumAddInplace('bsh,ho->bso')
        _test_fp32(model_v1, input1, input2, bias)
        
        input1 = torch.randn(8, 3, 768)
        input2 = torch.randn(768, 2304)
        model = EinsumAddScalar('bsh,ho->bso').eval()
        res_ref = model(input1, input2)
        tr_model = torch.jit.trace(model, (input1, input2))
        tr_model = torch.jit.freeze(tr_model)
        tr_model(input1, input2)
        tr_model(input1, input2)
        trace_graph = tr_model.graph_for(input1, input2)
        res_jit = tr_model(input1, input2)
        self.assertEqual(res_ref, res_jit, prec=1e-3)
        self.assertTrue(any(n.kind() == "ipex::einsum_binary" for n in trace_graph.nodes()))

        bias = torch.randn(4, 3, 2304)
        input1 = torch.randn(4, 3, 768)
        input2 = torch.randn(768, 2304)
        model_v1 = EinsumAddInplaceV1('bsh,ho->bso')
        _test_fp32(model_v1, input1, input2, bias, kind_in_graph='aten::einsum')

        bias1 = torch.randn(2, 4, 128, 128)
        input3 = torch.randn(2, 4, 128, 768)
        input4 = torch.randn(2, 4, 128, 768)
        model_v2 = EinsumAdd("bnqd,bnkd->bnqk")
        _test_fp32(model_v2, input3, input4, bias1)
        
        bias1 = torch.randn(8, 1, 1, 128)
        input3 = torch.randn(8, 4, 128, 768)
        input4 = torch.randn(8, 4, 128, 768)
        model_v2 = EinsumAdd("bnqd,bnkd->bnqk")
        _test_fp32(model_v2, input3, input4, bias1)

        bias1 = torch.randn(2, 4, 128, 768)
        input1 = torch.randn(2, 4, 128, 768)
        input2 = torch.randn(4, 768, 768)
        model_v2 = EinsumAdd("balh,ahr->balr")
        _test_fp32(model_v2, input1, input2, bias1)

        bias1 = torch.randn(768)
        input1 = torch.randn(128, 1024)
        input2 = torch.randn(768, 1024)
        model_v2 = EinsumAdd("mc,nc->mn")
        _test_fp32(model_v2, input1, input2, bias1)

        bias1 = torch.randn(768)
        input1 = torch.randn(128, 1024)
        input2 = torch.randn(1024, 768)
        model_v2 = EinsumAdd("mc,cn->mn")
        _test_fp32(model_v2, input1, input2, bias1)
        
        bias1 = torch.randn(1024)
        input1 = torch.randn(1024, 1024)
        input2 = torch.randn(1024, 1024)
        model_v2 = EinsumAdd("mc,cn->nm")
        _test_fp32(model_v2, input1, input2, bias1)
        
        bias1 = torch.randn(768)
        input1 = torch.randn(2, 128, 1024)
        input2 = torch.randn(1024, 23, 768)
        model_v2 = EinsumAdd("bqc,chv->bqhv")
        _test_fp32(model_v2, input1, input2, bias1)
        
        bias = torch.randn(768)
        input1 = torch.randn(2, 128, 16, 64)
        input2 = torch.randn(16,64, 768)
        model = EinsumAdd("bqhc,hco->bqo")
        _test_fp32(model, input1, input2, bias)
        
        bias = torch.randn(8)
        input1 = torch.randn(8)
        input2 = torch.randn(8)
        model = EinsumAdd("i,i->")
        _test_fp32(model, input1, input2, bias)
       
        #the output of torch.einsum("ij,j") is tensor([]) 
        bias = torch.randn(1)
        input1 = torch.randn(0, 3) 
        input2 = torch.randn(3)
        model = EinsumAdd(("ij,j"))
        _test_fp32(model, input1, input2, bias)

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
#use torch.sigmoid_(x)
                eltwise_fn = getattr(torch, eltwise_fn_name)
                m = M(eltwise_fn)
            elif eltwise == 'clamp':
                eltwise_fn = getattr(torch, eltwise_fn_name)
                m = M(eltwise_fn, {"min": 0, "max": 2})
            else:
#use F.elu(x, inplace = True)
                eltwise_fn = getattr(F, eltwise)
                m = M(eltwise_fn, {"inplace": True})

            with torch.no_grad():
                m.eval()
                x = torch.randn(1, 3, 16, 16)

#test restore inplace
                traced = torch.jit.trace(m, x)
                trace_graph = traced.graph_for(x)
                self.assertTrue(any(n.kind() == "aten::" + eltwise_fn_name for n in trace_graph.nodes()))

                y = m(x)
                traced_y = traced(x)
                self.assertEqual(y, traced_y)


    def test_enable_inplace(self):
        # M_apply_inplace is for testing success inplace replacement condition
        class M_apply_inplace(nn.Module):
            def __init__(self, eltwise_fn, params_dict={}):
                super(M_apply_inplace, self).__init__()
                self.eltwise = eltwise_fn
                self.params_dict = params_dict

            def forward(self, x):
                # put a softmax here for following reasons:
                # (1) x is the input, pass it to eltwise op will make it unable to be inplace
                # (2) ipex::softmax will not be fused into TE with following eltwise
                x1 = nn.Softmax(dim=-1)(x)
                x1 = self.eltwise(x1, **self.params_dict)
                return x1

        # M_remain_outplace is for testing failed inplace replacement condition
        class M_remain_outplace(nn.Module):
            def __init__(self, eltwise_fn, params_dict={}):
                super(M_remain_outplace, self).__init__()
                self.eltwise = eltwise_fn
                self.params_dict = params_dict

            def forward(self, x):
                x1 = self.eltwise(x, **self.params_dict)
                return x1

        for eltwise in ['sigmoid', 'tanh', 'celu', 'elu', 'hardsigmoid', 'hardswish', 'hardtanh', 'leaky_relu', 'relu6', 'relu', 'rrelu', 'selu', 'silu']:
            eltwise_fn_name = eltwise + '_'
            if eltwise in ['sigmoid', 'tanh', 'celu', 'relu', 'rrelu', 'selu']:
                eltwise_fn_outplace = getattr(torch, eltwise)
                m_inplace = M_apply_inplace(eltwise_fn_outplace)
                m_outplace = M_remain_outplace(eltwise_fn_outplace)
            else:
                eltwise_fn = getattr(F, eltwise)
                m_inplace = M_apply_inplace(eltwise_fn)
                m_outplace = M_remain_outplace(eltwise_fn)

            with torch.no_grad():
                m_inplace.eval()
                m_outplace.eval()
                x = torch.randn(1, 3, 16, 16)

                traced_inplace = torch.jit.trace(m_inplace, x)
                trace_graph_inplace = traced_inplace.graph_for(x)
                self.assertTrue(any(n.kind() == "aten::" + eltwise_fn_name for n in trace_graph_inplace.nodes()))
                y_inplace = m_inplace(x)
                traced_y_inplace = traced_inplace(x)
                self.assertEqual(y_inplace, traced_y_inplace)

                traced_outplace = torch.jit.trace(m_outplace, x)
                trace_graph_outplace = traced_outplace.graph_for(x)
                self.assertTrue(any(n.kind() == "aten::" + eltwise for n in trace_graph_outplace.nodes()))
                y_outplace = m_outplace(x)
                traced_y_outplace = traced_outplace(x)
                self.assertEqual(y_outplace, traced_y_outplace)

    @skipIfNoTorchVision
    def test_conv_torchvision_bn_folding(self):
        from torchvision.ops import misc as misc_nn_ops
        class M(nn.Module):
            def __init__(self):
                super(M, self).__init__()
                norm_layer = misc_nn_ops.FrozenBatchNorm2d
                self.inplanes = 64
                self.dilation = 1
                self.groups = 1
                self.base_width = 64
                self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = norm_layer(self.inplanes)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                return x

        model = M().eval()
        self._test_output(
            model,
            torch.randn(1, 3, 1200, 1200),
            kind_in_graph="ipex_prepack::convolution_relu_run",
            kind_not_in_graph="aten::add")

        self._test_output(
            model,
            torch.randn(1, 3, 1200, 1200),
            kind_in_graph="ipex_prepack::convolution_relu_run",
            kind_not_in_graph="aten::mul")

        self._test_output_bf16(
            model,
            torch.randn(1, 3, 1200, 1200),
            kind_in_graph="ipex_prepack::convolution_relu_run",
            kind_not_in_graph="aten::add",
            prec=0.1)

        self._test_output_bf16(
            model,
            torch.randn(1, 3, 1200, 1200),
            kind_in_graph="ipex_prepack::convolution_relu_run",
            kind_not_in_graph="aten::mul",
            prec=0.1)

    def test_TEfusion_with_dynamic_input(self):
        model = AddMulDiv().eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(model, torch.randn(11, 3, 20, 20)).eval()
            traced_model = torch.jit.freeze(traced_model)

        for i in range(5):
            input = torch.randn(i, 3, 20, 20)
            tresult = traced_model(input)
            result = model(input)
            self.assertEqual(tresult, result)

    def test_lru_cache_resize(self):
        import os
        # Set LRU_CACHE_CAPACITY < 1024 to trigger resize
        os.environ["LRU_CACHE_CAPACITY"] = "512"
        self.test_conv_fusion()
        os.environ["LRU_CACHE_CAPACITY"] = "1024"

if __name__ == '__main__':
    torch.manual_seed(2020)
    test = unittest.main()
