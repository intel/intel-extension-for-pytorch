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

import math
import sys
import tempfile
import unittest

from copy import deepcopy
from functools import reduce
from itertools import product
from operator import mul
from math import pi


import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import _Reduction
from common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
    TEST_WITH_ROCM
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors
from torch.autograd import Variable
import torch.backends.cudnn


# tarfile module tries to obtain a file object name in python 3.3
if sys.version_info[:2] == (3, 3):
    TemporaryFile = tempfile.NamedTemporaryFile
else:
    TemporaryFile = tempfile.TemporaryFile
PRECISION = 1e-5


def get_reduction(m):
    result = getattr(m, 'reduction', None)
    if result is None:
        result = _Reduction.legacy_get_string(getattr(m, 'sizeAverage', None), True, emit_warning=False)
    assert result is not None
    return result


def get_weight(m):
    result = getattr(m, 'weight', None)
    if result is not None:
        return result
    return getattr(m, 'weights', None)

module_tests = [
    dict(
        module_name='Linear',
        constructor_args=(10, 8),
        input_size=(4, 10),
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8),
    ),
    dict(
        module_name='Linear',
        constructor_args=(10, 8, False),
        input_size=(4, 10),
        desc='no_bias',
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t())
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2., 1.),
        input_size=(2, 3, 4, 5),
        check_inplace=True,
        desc='threshold_value'
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2., 10.),
        input_size=(2, 3, 4, 5),
        desc='large_value'
    ),
    dict(
        module_name='ReLU',
        input_size=(2, 3, 4, 5),
        check_inplace=True,
    ),
    dict(
        module_name='ReLU6',
        input_size=(2, 3, 4, 5),
        check_inplace=True,
    ),
    dict(
        module_name='RReLU',
        input_size=(1, 2, 2),
        test_cuda=False,
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        input_size=(4, 4, 5),
        desc='with_up_down',
        test_cuda=False,
    ),
    dict(
        module_name='Hardtanh',
        input_size=(3, 2, 5),
        reference_fn=lambda i, *_: i.clamp(-1, 1),
    ),
    dict(
        module_name='Sigmoid',
        input_size=(2, 3, 4, 5)
    ),
    dict(
        module_name='Tanh',
        input_size=(2, 3, 4, 5)
    ),
    dict(
        module_name='Flatten',
        input_size=(2, 3, 4, 5),
        reference_fn=lambda i, *_: torch.flatten(i, 1)
    ),
    dict(
        module_name='Softmax',
        constructor_args=(1,),
        input_size=(10, 20),
        reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20)),
    ),
    dict(
        module_name='Softmax2d',
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(1, False)),
    ),
    dict(
        module_name='LogSoftmax',
        constructor_args=(1,),
        input_size=(10, 20),
        reference_fn=lambda i, *_: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_(),
    ),
    dict(
        module_name='LogSoftmax',
        constructor_args=(1,),
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i, *_: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(),
        desc='multiparam',
    ),
    dict(
        module_name='ELU',
        constructor_args=(2.,),
        input_size=(3, 2, 5),
        reference_fn=lambda x, *_: torch.where(x >= 0, x, 2 * (x.exp() - 1)),
    ),
    # TODO: reference function
    dict(
        module_name='Hardshrink',
        constructor_args=(2.,),
        input_size=(4, 3, 2, 4),
    ),
    dict(
        module_name='LeakyReLU',
        input_size=(3, 2, 5),
        check_inplace=True
    ),
    dict(
        module_name='LeakyReLU',
        constructor_args=(0.5,),
        input_size=(3, 2, 5),
        check_inplace=True,
        desc='with_negval'
    ),
    dict(
        module_name='LogSigmoid',
        input_size=(2, 3, 4),
        reference_fn=lambda i, *_: i.sigmoid().log(),
    ),
    dict(
        module_name='Softplus',
        input_size=(10, 20),
        reference_fn=lambda i, *_: torch.log(1 + torch.exp(i)),
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2,),
        input_size=(10, 20),
        reference_fn=lambda i, *_: 1. / 2. * torch.log(1 + torch.exp(2 * i)),
        desc='beta',
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2, -100),
        input_size=(10, 20),
        reference_fn=(
            lambda i, *_: ((i * 2) > -100).type_as(i) * i
            + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log(1 + torch.exp(2 * i))
        ),
        desc='beta_threshold',
    ),
    dict(
        module_name='Softshrink',
        input_size=(3, 2, 5),
    ),
    dict(
        module_name='Softshrink',
        constructor_args=(1,),
        input_size=(3, 2, 5),
        desc='lambda',
    ),
    dict(
        module_name='CrossMapLRN2d',
        constructor_args=(5, 5e-3, 1e-3, 2),
        input_size=(2, 3, 6, 6),
        check_gradgrad=False,
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4),
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='1d',
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4),
        desc='1d_multiparam',
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4, 5),
        desc='2d',
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4, 5),
        desc='2d_multiparam',
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4, 5, 6),
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='3d',
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4, 5, 6),
        desc='3d_multiparam',
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='Softsign',
        input_size=(3, 2, 5),
        reference_fn=lambda i, *_: i.div(1 + torch.abs(i)),
    ),
    dict(
        module_name='Softmin',
        constructor_args=(1,),
        input_size=(10, 20),
    ),
    dict(
        module_name='Softmin',
        constructor_args=(1,),
        input_size=(2, 3, 5, 10),
        desc='multidim',
    ),
    dict(
        module_name='Tanhshrink',
        input_size=(2, 3, 4, 5),
    ),
]


# Generates rand tensor with non-equal values. This ensures that duplicate
# values won't be causing test failure for modules like MaxPooling.
# size should be small, otherwise randperm fails / long overflows.
def _rand_tensor_non_equal(*size):
    total = reduce(mul, size, 1)
    return torch.randperm(total).view(*size).double()


def wrap_functional(fn, **kwargs):
    class FunctionalModule(nn.Module):
        def forward(self, *args):
            return fn(*args, **kwargs)
    return FunctionalModule


def poissonnllloss_no_reduce_test():
    t = torch.randn(10, 10)
    return dict(
        fullname='PoissonNLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(10, 10),
        reference_fn=lambda i, *_: i.exp() - t.mul(i),
        pickle=False)


def bceloss_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    return dict(
        fullname='BCELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False,
        precision=7e-4)


def bceloss_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).double()
    return dict(
        fullname='BCELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False)


def bceloss_weights_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    weights = torch.rand(10)
    return dict(
        fullname='BCELoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, p, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False,
        precision=3e-4
    )


def bceloss_weights_no_reduce_scalar_test():
    t = torch.randn(()).double()
    weights = torch.rand(())
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False
    )


def bce_with_logistic_legacy_enum_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_legacy_enum',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
    )


def bce_with_logistic_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False,
    )


def bce_with_logistic_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).double()
    sigmoid = nn.Sigmoid()
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,
        pickle=False
    )


def kldivloss_with_target_no_reduce_test():
    i = torch.rand(10, 10).log()
    return dict(
        fullname='KLDivLoss_with_target_no_reduce',
        constructor=wrap_functional(
            lambda t: F.kl_div(i.type_as(t), t, reduction='none')),
        input_fn=lambda: torch.rand(10, 10),
        reference_fn=lambda t, *_:
            loss_reference_fns['KLDivLoss'](i.type_as(t), t, reduction='none'),
        pickle=False)


def kldivloss_no_reduce_test():
    t = torch.randn(10, 10)
    return dict(
        fullname='KLDivLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(10, 10).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        pickle=False,
    )


def kldivloss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='KLDivLoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.rand(()).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        pickle=False)


def l1loss_no_reduce_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='L1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(2, 3, 4),
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        pickle=False)


def l1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='L1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(()),
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        pickle=False)


def mseloss_no_reduce_test():
    input_size = (2, 3, 4, 5)
    target = torch.randn(*input_size)
    return dict(
        fullname='MSELoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),
        input_size=input_size,
        reference_fn=lambda i, *_: (i - target).pow(2),
        pickle=False)


def mseloss_no_reduce_scalar_test():
    input_size = ()
    target = torch.randn(input_size)
    return dict(
        fullname='MSELoss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),
        input_size=input_size,
        reference_fn=lambda i, *_: (i - target).pow(2),
        pickle=False)


def nllloss_no_reduce_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(15, 10).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss_no_reduce_ignore_index_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    kwargs = {'ignore_index': 2, 'reduction': 'none'}
    return dict(
        fullname='NLLLoss_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(15, 10).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss_no_reduce_weights_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLoss_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss_no_reduce_weights_ignore_index_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': 2}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i.data))),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss_no_reduce_weights_ignore_index_neg_test():
    t = Variable(torch.Tensor(15).uniform_().mul(10).floor().long())
    weight = torch.rand(10)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': -1}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index_neg',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input=torch.rand(15, 10).add(1e-2).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nllloss2d_no_reduce_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss2d_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    kwargs = {'ignore_index': 1, 'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nllloss2d_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLoss2d_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def nlllossNd_no_reduce_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLossNd_no_reduce',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nlllossNd_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs = {'ignore_index': 1, 'reduction': 'none'}
    return dict(
        fullname='NLLLossNd_no_reduce_ignore_index',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs)),
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False)


def nlllossNd_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLossNd_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False)


def smoothl1loss_no_reduce_test():
    t = torch.randn(2, 3, 4)
    return dict(
        fullname='SmoothL1Loss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(2, 3, 4),
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),
        pickle=False)


def smoothl1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(
        fullname='SmoothL1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(()),
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),
        pickle=False)


def multilabelmarginloss_1d_no_reduce_test():
    t = Variable(torch.rand(10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_index_neg_test():
    t = Variable(torch.clamp(torch.rand(5, 10).add(-.5).mul(20).floor().long(), min=-1))
    return dict(
        fullname='MultiLabelMarginLoss_index_neg',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multilabelmarginloss_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(10).floor().long())
    return dict(
        fullname='MultiLabelMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def hingeembeddingloss_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).double().mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), reduction='none'),
        check_sum_reduction=True,
        pickle=False)


def hingeembeddingloss_margin_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).double().mul_(2).sub(1))
    return dict(
        fullname='HingeEmbeddingLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), margin=0.5, reduction='none')),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), margin=0.5, reduction='none'),
        check_sum_reduction=True,
        pickle=False)


def softmarginloss_no_reduce_test():
    t = torch.randn(5, 5)
    return dict(
        fullname='SoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.soft_margin_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(5, 5),
        reference_fn=lambda i, *_:
            loss_reference_fns['SoftMarginLoss'](i, t.type_as(i), reduction='none'),
        pickle=False)


def multilabelsoftmarginloss_no_reduce_test():
    t = torch.rand(5, 10).mul(2).floor()
    return dict(
        fullname='MultiLabelSoftMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log())).sum(dim=1) / i.size(1),
        check_gradgrad=False,
        pickle=False)


def multilabelsoftmarginloss_weights_no_reduce_test():
    t = torch.rand(5, 10).mul(2).floor()
    weights = torch.rand(10)
    return dict(
        fullname='MultiLabelSoftMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i),
                                                    weight=weights.type_as(i), reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * weights).sum(dim=1) / i.size(1),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_1d_no_reduce_test():
    t = torch.rand(1).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_1d_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        input_fn=lambda: torch.randn(10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_p_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_p_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), p=2, reduction='none')),
        input_fn=lambda: torch.randn(5, 10).clamp_(1e-2, 1 - 1e-2),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), p=2, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_margin_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_margin_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), margin=0.5, reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  margin=0.5, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def multimarginloss_weights_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    weights = torch.rand(10)
    return dict(
        fullname='MultiMarginLoss_weights_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), weight=weights.type_as(i),
                                          reduction='none')),
        input_fn=lambda: torch.randn(5, 10),
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  weight=weights, reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False)


def fractional_max_pool2d_test(test_case):
    random_samples = torch.DoubleTensor(1, 3, 2).uniform_()
    if test_case == 'ratio':
        return dict(
            constructor=lambda: nn.FractionalMaxPool2d(
                2, output_ratio=0.5, _random_samples=random_samples),
            input_size=(1, 3, 5, 7),
            fullname='FractionalMaxPool2d_ratio')
    elif test_case == 'size':
        return dict(
            constructor=lambda: nn.FractionalMaxPool2d((2, 3), output_size=(
                4, 3), _random_samples=random_samples),
            input_size=(1, 3, 7, 6),
            fullname='FractionalMaxPool2d_size')


def fractional_max_pool3d_test(test_case):
    random_samples = torch.DoubleTensor(2, 4, 3).uniform_()
    if test_case == 'ratio':
        return dict(
            constructor=lambda: nn.FractionalMaxPool3d(
                2, output_ratio=0.5, _random_samples=random_samples),
            input_size=(2, 4, 5, 5, 5),
            fullname='FractionalMaxPool3d_ratio')
    elif test_case == 'size':
        return dict(
            constructor=lambda: nn.FractionalMaxPool3d((2, 2, 2), output_size=(
                4, 4, 4), _random_samples=random_samples),
            input_size=(2, 4, 7, 7, 7),
            fullname='FractionalMaxPool3d_size')
    elif test_case == 'asymsize':
        return dict(
            constructor=lambda: nn.FractionalMaxPool3d((4, 2, 3), output_size=(
                10, 3, 2), _random_samples=random_samples),
            input_size=(2, 4, 16, 7, 5),
            fullname='FractionalMaxPool3d_asymsize')


new_module_tests = [
    poissonnllloss_no_reduce_test(),
    bceloss_no_reduce_test(),
    bceloss_weights_no_reduce_test(),
    bce_with_logistic_legacy_enum_test(),
    bce_with_logistic_no_reduce_test(),
    bceloss_no_reduce_scalar_test(),
    bceloss_weights_no_reduce_scalar_test(),
    bce_with_logistic_no_reduce_scalar_test(),
    kldivloss_with_target_no_reduce_test(),
    kldivloss_no_reduce_test(),
    kldivloss_no_reduce_scalar_test(),
    l1loss_no_reduce_test(),
    l1loss_no_reduce_scalar_test(),
    mseloss_no_reduce_test(),
    mseloss_no_reduce_scalar_test(),
    nllloss_no_reduce_test(),
    nllloss_no_reduce_ignore_index_test(),
    nllloss_no_reduce_weights_test(),
    nllloss_no_reduce_weights_ignore_index_test(),
    nllloss_no_reduce_weights_ignore_index_neg_test(),
    nllloss2d_no_reduce_test(),
    nllloss2d_no_reduce_weights_test(),
    nllloss2d_no_reduce_ignore_index_test(),
    nlllossNd_no_reduce_test(),
    nlllossNd_no_reduce_weights_test(),
    nlllossNd_no_reduce_ignore_index_test(),
    smoothl1loss_no_reduce_test(),
    smoothl1loss_no_reduce_scalar_test(),
    multilabelmarginloss_1d_no_reduce_test(),
    multilabelmarginloss_index_neg_test(),
    multilabelmarginloss_no_reduce_test(),
    hingeembeddingloss_no_reduce_test(),
    hingeembeddingloss_margin_no_reduce_test(),
    softmarginloss_no_reduce_test(),
    multilabelsoftmarginloss_no_reduce_test(),
    multilabelsoftmarginloss_weights_no_reduce_test(),
    multimarginloss_no_reduce_test(),
    multimarginloss_1d_no_reduce_test(),
    multimarginloss_p_no_reduce_test(),
    multimarginloss_margin_no_reduce_test(),
    multimarginloss_weights_no_reduce_test(),
    fractional_max_pool2d_test('ratio'),
    fractional_max_pool2d_test('size'),
    fractional_max_pool3d_test('ratio'),
    fractional_max_pool3d_test('size'),
    fractional_max_pool3d_test('asymsize'),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10,),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='affine',
        test_cuda=(not TEST_WITH_ROCM),
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(5,),
        input_size=(4, 5, 3),
        cudnn=True,
        check_eval=True,
        desc='3d_input',
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10, 1e-3, None),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='affine_simple_average',
        test_cuda=(not TEST_WITH_ROCM),
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10, 1e-3, 0.3, False),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='not_affine',
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(10, 1e-3, 0.3, True, False),
        input_size=(4, 10),
        cudnn=True,
        check_eval=True,
        desc='not_tracking_stats',
        test_cuda=(not TEST_WITH_ROCM),
    ),
    dict(
        module_name='BatchNorm1d',
        constructor_args=(5, 1e-3, 0.3, False),
        input_size=(4, 5, 3),
        cudnn=True,
        check_eval=True,
        desc='3d_input_not_affine',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3,),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, None),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='2d_simple_average',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='momentum',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8, False),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='not_affine',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8, True, False),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='not_tracking_stats',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3,),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, None),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='3d_simple_average',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='momentum',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7, False),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='not_affine',
    ),
    dict(
        module_name='BatchNorm3d',
        constructor_args=(3, 1e-3, 0.7, True, False),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='not_tracking_stats',
    ),
    dict(
        module_name='InstanceNorm1d',
        constructor_args=(3, 1e-3, 0.3),
        input_size=(4, 3, 15),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='InstanceNorm1d',
        constructor_args=(3, 1e-3, 0.3, False, True),
        input_size=(4, 3, 15),
        cudnn=True,
        check_eval=True,
        desc='tracking_stats',
    ),
    dict(
        module_name='InstanceNorm2d',
        constructor_args=(3, 1e-3, 0.3),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='InstanceNorm2d',
        constructor_args=(3, 1e-3, 0.3, False, True),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        check_eval=True,
        desc='tracking_stats',
    ),
    dict(
        module_name='InstanceNorm3d',
        constructor_args=(3, 1e-3, 0.3),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
    ),
    dict(
        module_name='InstanceNorm3d',
        constructor_args=(3, 1e-3, 0.3, False, True),
        input_size=(2, 3, 4, 4, 4),
        cudnn=True,
        check_eval=True,
        desc='tracking_stats',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([5], 1e-3),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([5], 1e-3, False),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_no_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([2, 2, 5], 1e-3),
        input_size=(4, 2, 2, 5),
        cudnn=True,
        check_eval=True,
        desc='3d_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([2, 2, 5], 1e-3, False),
        input_size=(4, 2, 2, 5),
        cudnn=True,
        check_eval=True,
        desc='3d_no_elementwise_affine',
    ),
    dict(
        module_name='LayerNorm',
        constructor_args=([5], 1e-3),
        input_size=(0, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_empty_elementwise_affine',
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(3, 6, 1e-3),
        input_size=(4, 6, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_affine',
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(5, 5, 1e-3, False),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_no_affine_IN',  # this setting is equivalent with InstanceNormi
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(1, 5, 1e-3, False),
        input_size=(4, 5, 5),
        cudnn=True,
        check_eval=True,
        desc='1d_no_affine_LN',  # this setting is equivalent with LayerNorm
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(3, 6, 1e-3),
        input_size=(4, 6, 2, 3),
        cudnn=True,
        check_eval=True,
        desc='2d_affine',
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(3, 3, 1e-3, False),
        input_size=(4, 3, 2, 3),
        cudnn=True,
        check_eval=True,
        desc='2d_no_affine_IN',  # this setting is equivalent with InstanceNorm
    ),
    dict(
        module_name='GroupNorm',
        constructor_args=(1, 3, 1e-3, False),
        input_size=(4, 3, 2, 3),
        cudnn=True,
        check_eval=True,
        desc='2d_no_affine_LN',  # this setting is equivalent with LayerNorm
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        input_size=(2, 4, 10),
        cudnn=True,
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3, 2),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='stride',
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3, 1, 1),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='pad1',
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 5, 1, 2),
        input_size=(2, 4, 10),
        cudnn=True,
        desc='pad2',
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 4, 3, 1, 1),
        input_size=(1, 4, 1),
        cudnn=True,
        desc='pad1size1',
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 4, 5, 1, 2),
        input_size=(1, 4, 1),
        cudnn=True,
        desc='pad2size1',
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        input_size=(0, 4, 10),
        cudnn=True,
        desc='zero_batch',
        test_cuda=(not TEST_WITH_ROCM),
    ),
    dict(
        fullname='Conv1d_dilated',
        constructor=lambda: nn.Conv1d(4, 5, kernel_size=3, dilation=2),
        input_size=(2, 4, 10),
    ),
    dict(
        fullname='Conv1d_groups',
        constructor=lambda: nn.Conv1d(4, 6, kernel_size=3, groups=2),
        input_size=(2, 4, 6),
        cudnn=True,
    ),
    dict(
        fullname='ConvTranspose1d',
        constructor=lambda: nn.ConvTranspose1d(3, 4, kernel_size=3, stride=(3,), padding=1, output_padding=(1,)),
        cudnn=True,
        input_size=(1, 3, 7),
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, False),
        input_size=(1, 3, 6),
        cudnn=True,
        desc='no_bias',
    ),
    dict(
        module_name='ConvTranspose1d',
        constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2),
        input_size=(1, 3, 6),
        cudnn=True,
        desc='dilated',
    ),
    dict(
        fullname='ConvTranspose1d_groups',
        constructor=lambda: nn.ConvTranspose1d(4, 6, 3, stride=(3,), padding=1, output_padding=(1,), groups=2),
        cudnn=True,
        input_size=(2, 4, 7),
    ),
    dict(
        module_name='MaxPool1d',
        constructor_args=(4,),
        cpp_constructor_args='(4)',
        input_size=(2, 10, 4),
    ),
    dict(
        module_name='MaxPool1d',
        constructor_args=(4, 4),
        cpp_constructor_args='(torch::nn::MaxPool1dOptions(4).stride(4))',
        input_size=(2, 10, 4),
        desc='stride',
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2)),
        input_size=(2, 3, 7, 5),
        cudnn=True,
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2)),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='strided',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),
        input_size=(2, 3, 6, 6),
        cudnn=True,
        desc='padding',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),
        input_size=(2, 3, 8, 8),
        cudnn=True,
        desc='dilated',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2), 1, 0, 1, 1, False),
        input_size=(2, 3, 6, 5),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 2)),
        input_size=(0, 3, 7, 5),
        cudnn=True,
        desc='zero_batch',
        check_with_long_tensor=True,
        test_cuda=(not TEST_WITH_ROCM),
    ),
    dict(
        fullname='Conv2d_groups',
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
        input_size=(2, 4, 6, 5),
        cudnn=True,
        check_with_long_tensor=True,
    ),
    dict(
        fullname='Conv2d_groups_thnn',
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),
        input_size=(2, 4, 6, 5),
        check_with_long_tensor=True,
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (3, 2), 1, (1, 1)),
        cudnn=True,
        input_size=(1, 3, 7, 6),
        check_with_long_tensor=True,
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)),
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='dilated',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False),
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
    ),
    dict(
        fullname='ConvTranspose2d_groups',
        constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2),
        input_size=(1, 2, 4, 5),
        cudnn=True,
        check_with_long_tensor=True,
    ),
    dict(
        fullname='Conv2d_depthwise',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_with_multiplier',
        constructor=lambda: nn.Conv2d(4, 8, (3, 3), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_strided',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), stride=(2, 2), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_padded',
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), padding=(1, 1), groups=4),
        input_size=(2, 4, 6, 6),
    ),
    dict(
        fullname='Conv2d_depthwise_dilated',
        constructor=lambda: nn.Conv2d(4, 4, (2, 2), dilation=(2, 2), groups=4),
        input_size=(2, 4, 5, 5),
    ),
    dict(
        module_name='MaxPool2d',
        constructor_args=((3, 3), (2, 2), (1, 1)),
        cpp_constructor_args='(torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1}))',
        input_size=(1, 3, 7, 7),
        check_with_channels_last=True,
    ),
    dict(
        module_name='AvgPool1d',
        constructor_args=(2,),
        cpp_constructor_args="(2)",
        input_size=(2, 3, 6),
    ),
    dict(
        module_name='AvgPool1d',
        constructor_args=((2,), (2,)),
        cpp_constructor_args="(torch::nn::AvgPool1dOptions(2).stride(2))",
        input_size=(2, 3, 6),
        desc='stride',
    ),
    dict(
        module_name='AvgPool1d',
        constructor_args=(2, 2, 1),
        cpp_constructor_args="(torch::nn::AvgPool1dOptions(2).stride(2).padding(1))",
        input_size=(2, 3, 6),
        desc='stride_pad',
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2),),
        cpp_constructor_args="(torch::nn::AvgPool2dOptions({2, 2}))",
        input_size=(2, 3, 6, 6),
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2), (2, 2)),
        cpp_constructor_args="(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}))",
        input_size=(2, 3, 6, 6),
        desc='stride',
    ),
    dict(
        module_name='AvgPool2d',
        constructor_args=((2, 2), (2, 2), (1, 1)),
        cpp_constructor_args="(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).padding({1, 1}))",
        input_size=(2, 3, 6, 6),
        desc='stride_pad',
    ),
    dict(
        fullname='AvgPool2d_divisor',
        constructor=lambda: nn.AvgPool2d((2, 2), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool2dOptions({2, 2}).divisor_override(1))",
        input_size=(2, 3, 6, 6),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool2d_divisor_stride',
        constructor=lambda: nn.AvgPool2d((2, 2), (2, 2), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).divisor_override(1))",
        input_size=(2, 3, 6, 6),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool2d_divisor_stride_pad',
        constructor=lambda: nn.AvgPool2d((2, 2), (2, 2), (1, 1), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).padding({1, 1}).divisor_override(1))",
        input_size=(2, 3, 6, 6),
        check_with_long_tensor=True,
    ),
    dict(
        module_name='LPPool2d',
        constructor_args=(2, 2, 2),
        input_size=(1, 3, 7, 7),
    ),
    dict(
        module_name='LPPool2d',
        constructor_args=(1.5, 2),
        input_fn=lambda: torch.rand(1, 3, 7, 7),
        desc='norm',
    ),
    dict(
        module_name='LPPool1d',
        constructor_args=(1.5, 2),
        input_fn=lambda: torch.rand(1, 3, 7),
        desc='norm',
    ),
    dict(
        module_name='LPPool1d',
        constructor_args=(2, 2, 3),
        input_size=(1, 3, 7),
    ),
    dict(
        module_name='LocalResponseNorm',
        constructor_args=(3, ),
        input_size=(1, 5, 7),
        desc='1d',
    ),
    dict(
        module_name='LocalResponseNorm',
        constructor_args=(2, ),
        input_size=(1, 5, 7, 7),
        desc='2d_uneven_pad',
    ),
    dict(
        module_name='LocalResponseNorm',
        constructor_args=(1, 1., 0.5, 2.),
        input_size=(1, 5, 7, 7, 7),
        desc='3d_custom_params',
    ),
    dict(
        module_name='ReflectionPad1d',
        constructor_args=((1, 2),),
        input_size=(2, 3, 8),
    ),
    dict(
        module_name='ReflectionPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 8, 8),
    ),
    dict(
        module_name='ReplicationPad1d',
        constructor_args=((1, 2),),
        input_size=(2, 3, 4),
    ),
    dict(
        module_name='ReplicationPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 4, 4),
    ),
    dict(
        module_name='ZeroPad2d',
        constructor_args=((1, 2, 3, 4),),
        input_size=(2, 3, 4, 4)
    ),
    dict(
        module_name='ZeroPad2d',
        constructor_args=((-1, -1, -1, -2),),
        input_size=(2, 3, 4, 4),
        desc='negative_dims'
    ),
    dict(
        module_name='ConstantPad1d',
        constructor_args=((1, 2), 2.),
        input_size=(2, 3, 4)
    ),
    dict(
        module_name='ConstantPad2d',
        constructor_args=((1, 2, 3, 4), 2.),
        input_size=(2, 3, 4, 4)
    ),
    dict(
        module_name='ConstantPad3d',
        constructor_args=((1, 2, 3, 4, 1, 0), 2.),
        input_size=(2, 3, 4, 4, 5)
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, (2, 3, 4)),
        input_size=(2, 3, 3, 4, 5),
        cudnn=True,
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, (2, 3, 4), 1, 0, 1, 1, False),
        input_size=(2, 3, 3, 4, 5),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2),
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2, 1),
        input_size=(2, 3, 5, 5, 5),
        cudnn=True,
        desc='stride_padding',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, (2, 3, 4)),
        input_size=(0, 3, 3, 4, 5),
        cudnn=True,
        check_with_long_tensor=True,
        desc='zero_batch',
        test_cuda=(not TEST_WITH_ROCM),
    ),
    dict(
        fullname='Conv3d_groups',
        constructor=lambda: nn.Conv3d(4, 6, kernel_size=3, groups=2),
        input_size=(2, 4, 4, 5, 4),
        cudnn=True,
        check_with_long_tensor=True,
    ),
    dict(
        fullname='Conv3d_dilated',
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        fullname='Conv3d_dilated_strided',
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2, stride=2),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2)),
        cudnn=True,
        input_size=(1, 2, 4, 5, 4),
    ),
    dict(
        module_name='ConvTranspose3d',
        constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)),
        cudnn=True,
        input_size=(1, 2, 4, 5, 4),
        desc='dilated',
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=((2, 2, 2),),
        cpp_constructor_args='(torch::nn::MaxPool3dOptions({2, 2, 2}))',
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=(2, (2, 2, 2)),
        cpp_constructor_args='(torch::nn::MaxPool3dOptions(2).stride({2, 2, 2}))',
        input_size=(2, 3, 5, 5, 5),
        desc='stride',
    ),
    dict(
        module_name='MaxPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        cpp_constructor_args='(torch::nn::MaxPool3dOptions(2).stride(2).padding({1, 1, 1}))',
        input_size=(2, 3, 5, 5, 5),
        desc='stride_padding',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=((2, 2, 2),),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions({2, 2, 2}))",
        input_size=(2, 3, 4, 4, 4),
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, (2, 2, 2)),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(2).stride({2, 2, 2}))",
        input_size=(2, 3, 5, 5, 5),
        desc='stride',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1}))",
        input_size=(2, 3, 5, 5, 5),
        desc='stride_pad',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(4, 2, (1, 2, 1)),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(4).stride(2).padding({1, 2, 1}))",
        input_size=(2, 3, 5, 5, 5),
        desc='stride_pad_gpu_fixedkw_output',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=((2, 4, 8), 1, (1, 1, 2)),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions({2, 4, 8}).stride(1).padding({1, 1, 2}))",
        input_size=(2, 3, 2, 4, 8),
        desc='stride_pad_gpu_general_output',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(3, 1, 0),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(3).stride(1).padding(0))",
        input_size=(2, 3, 4, 4, 4),
        desc='stride1_pad0_gpu_input',
    ),
    dict(
        module_name='AvgPool3d',
        constructor_args=(2, 2, (1, 1, 1)),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1}))",
        input_size=(2, 3, 4, 4, 4),
        desc='stride_pad_gpu_input_nooverlap',
    ),
    dict(
        fullname='AvgPool3d_divisor',
        constructor=lambda: nn.AvgPool3d((2, 2, 2), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions({2, 2, 2}).divisor_override(1))",
        input_size=(2, 3, 4, 4, 4),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool3d_divisor_stride',
        constructor=lambda: nn.AvgPool3d(2, (2, 2, 2), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(2).stride({2, 2, 2}).divisor_override(1))",
        input_size=(2, 3, 5, 5, 5),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool3d_divisor_stride_pad',
        constructor=lambda: nn.AvgPool3d(2, 2, (1, 1, 1), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1}).divisor_override(1))",
        input_size=(2, 3, 5, 5, 5),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool3d_divisor_stride_pad_gpu_fixedkw_output',
        constructor=lambda: nn.AvgPool3d(4, 2, (1, 2, 1), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(4).stride(2).padding({1, 2, 1}).divisor_override(1))",
        input_size=(2, 3, 5, 5, 5),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool3d_divisor_stride_pad_gpu_general_output',
        constructor=lambda: nn.AvgPool3d((2, 4, 8), 1, (1, 1, 2), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions({2, 4, 8}).stride(1).padding({1, 1, 2}).divisor_override(1))",
        input_size=(2, 3, 2, 4, 8),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool3d_divisor_stride1_pad0_gpu_input',
        constructor=lambda: nn.AvgPool3d(3, 1, 0, divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(3).stride(1).padding(0).divisor_override(1))",
        input_size=(2, 3, 4, 4, 4),
        check_with_long_tensor=True,
    ),
    dict(
        fullname='AvgPool3d_divisor_stride_pad_gpu_input_nooverlap',
        constructor=lambda: nn.AvgPool3d(2, 2, (1, 1, 1), divisor_override=1),
        cpp_constructor_args="(torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1}).divisor_override(1))",
        input_size=(2, 3, 4, 4, 4),
        check_with_long_tensor=True,
    ),
    dict(
        module_name='ReplicationPad3d',
        constructor_args=((1, 2, 3, 4, 5, 6),),
        input_size=(2, 3, 5, 5, 5),
    ),
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        jacobian_input=False,
        check_gradgrad=False,
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        jacobian_input=False,
        check_gradgrad=False,
        desc='mean',
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3, None, 2., False, 'sum'),
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        jacobian_input=False,
        check_gradgrad=False,
        desc='sum',
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3, None, 2., False, 'max'),
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        jacobian_input=False,
        check_gradgrad=False,
        desc='max',
    ),
    dict(
        fullname='EmbeddingBag_sparse',
        constructor=lambda: nn.EmbeddingBag(4, 3, sparse=True),
        input_fn=lambda: torch.randperm(2).repeat(1, 2),
        jacobian_input=False,
        check_gradgrad=False,
    ),
    dict(
        constructor=lambda: nn.Embedding(4, 3, sparse=True),
        input_fn=lambda: torch.randperm(2).repeat(1, 2),
        jacobian_input=False,
        fullname='Embedding_sparse',
        check_gradgrad=False,
    ),
    dict(
        module_name='PixelShuffle',
        constructor_args=(3,),
        input_size=(1, 9, 4, 4),
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        input_size=(1, 2, 4),
        fullname='interpolate_nearest_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'),
        input_size=(1, 2, 3),
        fullname='interpolate_nearest_tuple_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        input_size=(1, 2, 4),
        fullname='interpolate_nearest_scale_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
        input_size=(1, 2, 4),
        fullname='interpolate_linear_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, ), scale_factor=None, mode='linear', align_corners=False),
        input_size=(1, 2, 3),
        fullname='interpolate_linear_tuple_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=False),
        input_size=(1, 2, 4),
        fullname='interpolate_linear_scale_1d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=True),
        input_size=(1, 2, 4),
        fullname='interpolate_linear_1d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=True),
        input_size=(1, 2, 4),
        fullname='interpolate_linear_scale_1d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=2, scale_factor=None, mode='nearest'),
        input_size=(1, 128, 1, 1),
        fullname='interpolate_nearest_2d_launch_configs',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_nearest_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, 16), scale_factor=None, mode='nearest'),
        input_size=(1, 2, 3, 4),
        fullname='interpolate_nearest_tuple_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_nearest_scale_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                    mode='bilinear', align_corners=False),
        input_size=(1, 2, 2, 3),
        fullname='interpolate_bilinear_tuple_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.,
                                    mode='bilinear', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bilinear', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_shared_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_skewed_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=True),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_tuple_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=True),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_skewed_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                    mode='bicubic', align_corners=False),
        input_size=(1, 2, 2, 3),
        fullname='interpolate_bicubic_tuple_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='bicubic', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bicubic', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_tuple_shared_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bicubic', align_corners=False),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_tuple_skewed_2d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bicubic', align_corners=True),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_tuple_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bicubic', align_corners=True),
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bicubic_scale_tuple_skewed_2d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        input_size=(1, 2, 4, 4, 4),
        fullname='interpolate_nearest_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, 16, 16), scale_factor=None, mode='nearest'),
        input_size=(1, 2, 3, 4, 4),
        fullname='interpolate_nearest_tuple_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        input_size=(1, 2, 4, 4, 4),
        fullname='interpolate_nearest_scale_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
        input_size=(1, 2, 4, 4, 4),
        fullname='interpolate_trilinear_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6, 6),
                                    scale_factor=None, mode='trilinear', align_corners=False),
        input_size=(1, 2, 2, 3, 3),
        fullname='interpolate_trilinear_tuple_3d',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=False),
        input_size=(1, 2, 3, 4, 4),
        fullname='interpolate_trilinear_scale_3d',
        # See https://github.com/pytorch/pytorch/issues/5006
        precision=3e-4,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6, 6), scale_factor=None,
                                    mode='trilinear', align_corners=True),
        input_size=(1, 2, 2, 3, 3),
        fullname='interpolate_trilinear_tuple_3d_align_corners',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=True),
        input_size=(1, 2, 3, 4, 4),
        fullname='interpolate_trilinear_scale_3d_align_corners',
        # See https://github.com/pytorch/pytorch/issues/5006
        precision=3e-4,
        pickle=False,
    ),
    dict(
        module_name='AdaptiveMaxPool1d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5),
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6),
        desc='single',
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=((3, 4),),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveMaxPool2d',
        constructor_args=((3, None),),
        input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6),
        desc='tuple_none',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7),
        desc='single',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=((3, 4, 5),),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=((3, None, 5),),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7),
        desc='tuple_none',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=(3,),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 12, 9, 3),
        desc='single_nonatomic',
    ),
    dict(
        module_name='AdaptiveMaxPool3d',
        constructor_args=((3, 4, 5),),
        input_fn=lambda: _rand_tensor_non_equal(2, 3, 6, 4, 10),
        desc='tuple_nonatomic',
    ),
    dict(
        module_name='AdaptiveAvgPool1d',
        constructor_args=(3,),
        input_fn=lambda: torch.rand(1, 3, 5),
    ),
    dict(
        module_name='AdaptiveAvgPool1d',
        constructor_args=(1,),
        input_fn=lambda: torch.rand(1, 3, 5),
        desc='one_output',
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=(3,),
        input_fn=lambda: torch.rand(1, 3, 5, 6),
        desc='single',
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=(1,),
        input_fn=lambda: torch.rand(1, 3, 5, 6),
        desc='single_1x1output',
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=((3, 4),),
        input_fn=lambda: torch.rand(1, 3, 5, 6),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveAvgPool2d',
        constructor_args=((3, None),),
        input_fn=lambda: torch.rand(1, 3, 5, 6),
        desc='tuple_none',
    ),
    dict(
        module_name='AdaptiveAvgPool3d',
        constructor_args=(3,),
        input_fn=lambda: torch.rand(2, 3, 5, 2, 7),
        desc='single',
    ),
    dict(
        module_name='AdaptiveAvgPool3d',
        constructor_args=((3, 4, 5),),
        input_fn=lambda: torch.rand(2, 3, 5, 3, 7),
        desc='tuple',
    ),
    dict(
        module_name='AdaptiveAvgPool3d',
        constructor_args=((None, 4, 5),),
        input_fn=lambda: torch.rand(2, 3, 5, 3, 7),
        desc='tuple_none',
    ),
    dict(
        module_name='SELU',
        input_size=(3, 2, 5),
        check_inplace=True
    ),
    dict(
        module_name='SELU',
        input_size=(),
        check_inplace=True,
        desc='scalar'
    ),
    dict(
        module_name='CELU',
        input_size=(3, 2, 5),
        constructor_args=(2.,),
        check_inplace=True,
        reference_fn=lambda x, *_: torch.where(x >= 0, x, 2. * ((.5 * x).exp() - 1)),
    ),
    dict(
        module_name='CELU',
        input_size=(),
        constructor_args=(2.,),
        check_inplace=True,
        reference_fn=lambda x, *_: torch.where(x >= 0, x, 2. * ((.5 * x).exp() - 1)),
        desc='scalar'
    ),
    dict(
        module_name='GLU',
        input_size=(5, 6),
    ),
    dict(
        module_name='GLU',
        constructor_args=(1,),
        input_size=(5, 6, 7),
        desc='dim',
    ),
    dict(
        module_name='GELU',
        input_size=(),
        desc='scalar',
        reference_fn=lambda x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
    ),
    dict(
        module_name='GELU',
        input_size=(3, 2, 5),
        reference_fn=lambda x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        input_size=(2, 128),
        fullname='softmax_lastdim_dtype',
        pickle=False,
        test_cuda=False
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='softmax_spatial_special',
        pickle=False,
        test_cuda=(not TEST_WITH_ROCM)
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='softmax_spatial_dtype',
        pickle=False,
        test_cuda=False
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=0),
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim0',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=3),
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim3',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        input_size=(),
        fullname='softmax_functional_scalar',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=-1),
        input_size=(2, 128),  # trigger the last-dim algo in CUDA
        fullname='log_softmax_lastdim',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        input_size=(2, 128, 2, 2),  # trigger special case of spatial CUDA algo
        fullname='log_softmax_spatial_special',
        pickle=False,
        test_cuda=(not TEST_WITH_ROCM)
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        input_size=(2, 2, 4, 4),  # regular spatial algorithm
        fullname='log_softmax_spatial',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=0),
        input_size=(2, 3, 4, 5),
        fullname='log_softmax_dim0',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=3),
        input_size=(2, 3, 4, 5),
        fullname='log_softmax_dim3',
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=0),
        input_size=(),
        fullname='log_softmax_scalar',
        pickle=False,
    ),
    dict(
        fullname='Unfold',
        constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)),
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold',
        constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
        cpp_constructor_args='(torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1}))',
        input_size=(2, 16, 4),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Unfold_int_input',
        constructor=lambda: nn.Unfold(2, 1, 0, 1),
        input_size=(2, 4, 3, 3),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        fullname='Fold_int_input',
        constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
        cpp_constructor_args='(torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1))',
        input_size=(2, 16, 4),
        check_gradgrad=False,
        test_cuda=True,
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2., 1.),
        input_size=(),
        check_inplace=True,
        desc='threshold_value_scalar'
    ),

    dict(
        module_name='ReLU',
        input_size=(),
        check_inplace=True,
        desc='scalar'
    ),
    dict(
        module_name='ReLU6',
        input_size=(),
        check_inplace=True,
        desc='scalar'
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        input_size=(),
        desc='with_up_down_scalar',
        test_cuda=False,
    ),
    dict(
        module_name='Hardtanh',
        input_size=(),
        reference_fn=lambda i, *_: i.clamp(-1, 1),
        desc='scalar'
    ),
    dict(
        module_name='Sigmoid',
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Tanh',
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Softmax',
        constructor_args=(0,),
        input_size=(),
        reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(0, True)),
        desc='scalar',
    ),
    dict(
        module_name='LogSoftmax',
        constructor_args=(0,),
        input_size=(),
        reference_fn=lambda i, *_: torch.exp(i).div_(torch.exp(i).sum(0, False)).log_(),
        desc='multiparam_scalar',
    ),
    dict(
        module_name='ELU',
        constructor_args=(2.,),
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Hardshrink',
        constructor_args=(2.,),
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='LeakyReLU',
        constructor_args=(0.5,),
        input_size=(),
        check_inplace=True,
        desc='with_negval_scalar'
    ),
    dict(
        module_name='LogSigmoid',
        input_size=(),
        reference_fn=lambda i, *_: i.sigmoid().log(),
        desc='scalar'
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2, -100),
        input_size=(),
        reference_fn=(
            lambda i, *_: ((i * 2) > -100).type_as(i) * i
            + ((i * 2) <= -100).type_as(i) * 1.0 / 2.0 * torch.log(1 + torch.exp(2 * i))
        ),
        desc='beta_threshold_scalar',
    ),
    dict(
        module_name='Softshrink',
        constructor_args=(1,),
        input_size=(),
        desc='lambda_scalar',
    ),
    dict(
        module_name='PReLU',
        input_size=(),
        reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='scalar',
    ),
    dict(
        module_name='Softsign',
        input_size=(),
        reference_fn=lambda i, *_: i.div(1 + torch.abs(i)),
        desc='scalar',
    ),
    dict(
        module_name='Softmin',
        constructor_args=(0,),
        input_size=(),
        desc='scalar',
    ),
    dict(
        module_name='Tanhshrink',
        input_size=(),
        desc='scalar',
    ),
    dict(
        fullname='Padding12_1dcircular',
        constructor=wrap_functional(F.pad, pad=(1, 2), mode='circular'),
        input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 2, 3]),
        reference_fn=lambda i, *_: padding1d_circular(i, (1, 2)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding31_1dcircular',
        constructor=wrap_functional(F.pad, pad=(3, 1), mode='circular'),
        input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 2, 3]),
        reference_fn=lambda i, *_: padding1d_circular(i, (3, 1)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding33_1dcircular',
        constructor=wrap_functional(F.pad, pad=(3, 3), mode='circular'),
        input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 2, 3]),
        reference_fn=lambda i, *_: padding1d_circular(i, (3, 3)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding1221_2dcircular',
        constructor=wrap_functional(F.pad, pad=(1, 2, 2, 1), mode='circular'),
        input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 1, 2, 3]),
        reference_fn=lambda i, *_: padding2d_circular(i, (1, 2, 2, 1)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding2322_2dcircular',
        constructor=wrap_functional(F.pad, pad=(2, 3, 2, 2), mode='circular'),
        input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 1, 2, 3]),
        reference_fn=lambda i, *_: padding2d_circular(i, (2, 3, 2, 2)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding3331_2dcircular',
        constructor=wrap_functional(F.pad, pad=(3, 3, 3, 1), mode='circular'),
        input_fn=lambda: torch.arange(9, out=torch.DoubleTensor()).reshape([1, 1, 3, 3]),
        reference_fn=lambda i, *_: padding2d_circular(i, (3, 3, 3, 1)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding122112_3dcircular',
        constructor=wrap_functional(F.pad, pad=(1, 2, 2, 1, 1, 2), mode='circular'),
        input_fn=lambda: torch.arange(12, out=torch.DoubleTensor()).reshape([1, 1, 2, 2, 3]),
        reference_fn=lambda i, *_: padding3d_circular(i, (1, 2, 2, 1, 1, 2)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding322112_3dcircular',
        constructor=wrap_functional(F.pad, pad=(3, 2, 2, 1, 1, 2), mode='circular'),
        input_fn=lambda: torch.arange(12, out=torch.DoubleTensor()).reshape([1, 1, 2, 2, 3]),
        reference_fn=lambda i, *_: padding3d_circular(i, (3, 2, 2, 1, 1, 2)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),
    dict(
        fullname='Padding332122_3dcircular',
        constructor=wrap_functional(F.pad, pad=(3, 3, 2, 1, 2, 2), mode='circular'),
        input_fn=lambda: torch.arange(12, out=torch.DoubleTensor()).reshape([1, 1, 2, 2, 3]),
        reference_fn=lambda i, *_: padding3d_circular(i, (3, 3, 2, 1, 2, 2)),
        skip_double=TEST_WITH_ROCM,
        pickle=False,
    ),

    dict(
        module_name='Conv1d',
        constructor_args=(3, 4, 2, 2, (1,), 1, 1, True, 'circular'),
        input_size=(2, 3, 5,),
        cudnn=True,
        desc='stride1_pad1circular',
    ),
    dict(
        module_name='Conv1d',
        constructor_args=(3, 4, 2, 2, (2,), 1, 1, True, 'circular'),
        input_size=(2, 3, 5,),
        cudnn=True,
        desc='stride1_pad2circular',
    ),
    dict(
        module_name='Conv2d',
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 2), 1, 1, True, 'circular'),
        input_size=(2, 3, 3, 3),
        cudnn=True,
        desc='pad2circular',
        check_with_long_tensor=True,
    ),
    dict(
        module_name='Conv3d',
        constructor_args=(3, 4, 2, 2, (1, 2, 3), 1, 1, True, 'circular'),
        input_size=(2, 3, 3, 3, 3),
        cudnn=True,
        desc='stride_pad1circular',
        check_with_long_tensor=True,
    ),
]


def kldivloss_reference(input, target, reduction='mean'):
    safe_target = target * (target > 0).type_as(target)
    safe_target_log = (safe_target + (target <= 0).type_as(target)).log()
    result = safe_target * (safe_target_log - input)
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    elif reduction == 'batchmean' and results.dim() != 0:
        return result.sum() / result.size(0)
    return result


def nlllossNd_reference(input, target, weight=None, ignore_index=-100,
                        reduction='mean'):
    assert input.dim() >= 3
    N = input.size(0)
    C = input.size(1)
    out_size = (N,) + input.size()[2:]
    output = torch.zeros(out_size).type_as(input)

    if weight is None:
        weight = torch.ones(C).type_as(input)
    total_weight = 0
    for tup in product(*[range(size) for size in out_size]):
        t_nx = target[tup]
        norm = 0. if ignore_index == t_nx else weight[t_nx].item()
        input_index = list(tup)
        input_index.insert(1, t_nx)
        output[tup] = -input[tuple(input_index)] * norm
        total_weight += norm

    if reduction == 'mean':
        return output.sum() / total_weight
    elif reduction == 'sum':
        return output.sum()
    return output


def nllloss_reference(input, target, weight=None, ignore_index=-100,
                      reduction='mean'):

    def nll_loss_helper(input, target, weight, ignore_index):
        if target == ignore_index:
            return (0, 0)
        norm = 1 if weight is None else weight[target]
        result = -input[target] * norm
        return (result, norm)

    losses_and_weights = [nll_loss_helper(i, t, weight, ignore_index)
                          for i, t in zip(input, target)]
    losses, weights = zip(*losses_and_weights)
    losses_tensor = input.new_tensor(losses)
    if reduction == 'mean':
        return sum(losses_tensor) / sum(weights)
    elif reduction == 'sum':
        return sum(losses_tensor)
    else:
        return losses_tensor


def smoothl1loss_reference(input, target, reduction='mean'):
    abs_diff = (input - target).abs()
    ge_one_mask = (abs_diff >= 1).type_as(abs_diff)
    lt_one_mask = (abs_diff < 1).type_as(abs_diff)
    output = ge_one_mask * (abs_diff - 0.5) + lt_one_mask * 0.5 * (abs_diff ** 2)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def _multilabelmarginloss_reference(input, target):
    targets = []
    for target_index in target:
        if target_index < 0:
            break
        targets.append(target_index)

    sum = 0
    for target_index in targets:
        for i in range(0, len(input)):
            if i not in targets:
                sum += max(0, 1 - input[target_index] + input[i])

    return sum


def multilabelmarginloss_reference(input, target, reduction='mean'):
    if input.dim() == 1:
        n = 1
        dim = input.size(0)
        output = input.new(n).zero_()
        output[0] = _multilabelmarginloss_reference(input, target)
    else:
        n = input.size(0)
        dim = input.size(1)
        output = input.new(n).zero_()
        for i in range(0, n):
            output[i] = _multilabelmarginloss_reference(input[i], target[i])

    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    return output / dim


def hingeembeddingloss_reference(input, target, margin=1.0, reduction='mean'):
    margin_clamp = (margin - input).clamp(min=0).type_as(input)
    output = torch.where(target == 1, input, margin_clamp)

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def softmarginloss_reference(input, target, reduction='mean'):
    output = (1 + (-input * target).exp()).log()

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def _multimarginloss_reference(input, target_idx, p, margin, weight):
    if weight is None:
        weight = input.new(len(input)).fill_(1)

    output = 0
    for i in range(0, len(input)):
        if i != target_idx:
            output += max(0, weight[target_idx] * (margin - input[target_idx] + input[i]) ** p)
    return output


def multimarginloss_reference(input, target, p=1, margin=1, weight=None, reduction='mean'):
    if input.dim() == 1:
        n = 1
        dim = input.size(0)
        return _multimarginloss_reference(input, target[0], p, margin, weight) / dim
    else:
        n = input.size(0)
        dim = input.size(1)
        output = input.new(n)
        for x in range(0, n):
            output[x] = _multimarginloss_reference(input[x], target[x], p, margin, weight)

        if reduction == 'mean':
            return output.mean() / dim
        elif reduction == 'sum':
            return output.sum() / dim
        return output / dim


def cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    def _cos(a, b):
        cos = a.new(a.size(0))
        for i in range(0, a.size(0)):
            cos[i] = (a[i] * b[i]).sum() / ((((a[i] * a[i]).sum() + 1e-12) * ((b[i] * b[i]).sum() + 1e-12)) ** 0.5)
        return cos

    output = torch.where(target == 1, 1 - _cos(input1, input2), (_cos(input1, input2) - margin).clamp(min=0))

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False,
                                reduction='mean'):
    d_p = torch.pairwise_distance(anchor, positive, p, eps)
    d_n = torch.pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = torch.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    output = torch.clamp(margin + d_p - d_n, min=0.0)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def marginrankingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    output = (-target * (input1 - input2) + margin).clamp(min=0)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


# this directly follows Graves et al's paper, in contrast to the production implementation, it does not use log-space
def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    dt = log_probs.dtype
    log_probs = log_probs.double()  # we need the accuracy as we are not in logspace
    targets = targets.long()
    cum_target_lengths = target_lengths.cumsum(0)
    losses = []
    for i in range(log_probs.size(1)):
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        probs = log_probs[:input_length, i].exp()
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        losses.append(-alpha[-2:].sum().log()[None])
    output = torch.cat(losses, 0)
    if reduction == 'mean':
        return (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        return output.sum()
    output = output.to(dt)
    return output


def padding1d_circular(input, pad):
    r""" input:
            [[[0., 1., 2.],
              [3., 4., 5.]]]
          pad: (1, 2)
          output:
            [[[2., 0., 1., 2., 0., 1.],
              [5., 3., 4., 5., 3., 4.]]]
    """
    return torch.cat([input[:, :, -pad[0]:], input,
                      input[:, :, 0:pad[1]]], dim=2)


def padding2d_circular(input, pad):
    r"""input:
             [[[[0., 1., 2],
                [3., 4., 5.]]]]
            pad: (1, 2, 2, 1)
    output:
        [[[[2., 0., 1., 2., 0., 1.],
           [5., 3., 4., 5., 3., 4.],
           [2., 0., 1., 2., 0., 1.],
           [5., 3., 4., 5., 3., 4.],
           [2., 0., 1., 2., 0., 1.]]]]
    """
    input = torch.cat([input[:, :, -pad[2]:], input, input[:, :, 0:pad[3]]], dim=2)
    return torch.cat([input[:, :, :, -pad[0]:], input, input[:, :, :, 0:pad[1]]], dim=3)


def padding3d_circular(input, pad):
    r"""input:
            [[[[[ 0.,  1.,  2.],
                [ 3.,  4.,  5.]],
               [[ 6.,  7.,  8.],
                [ 9., 10., 11.]]]]]
        pad: (1, 2, 2, 1, 1, 2)
        output: [[[[[ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.]],

              [[ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.]],

              [[ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.]],

              [[ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.]],

              [[ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.]]]]]
    """
    input = torch.cat([input[:, :, -pad[4]:], input, input[:, :, 0:pad[5]]], dim=2)
    input = torch.cat([input[:, :, :, -pad[2]:], input, input[:, :, :, 0:pad[3]]], dim=3)
    return torch.cat([input[:, :, :, :, -pad[0]:], input, input[:, :, :, :, 0:pad[1]]], dim=4)


loss_reference_fns = {
    'KLDivLoss': kldivloss_reference,
    'NLLLoss': nllloss_reference,
    'NLLLossNd': nlllossNd_reference,
    'SmoothL1Loss': smoothl1loss_reference,
    'MultiLabelMarginLoss': multilabelmarginloss_reference,
    'HingeEmbeddingLoss': hingeembeddingloss_reference,
    'SoftMarginLoss': softmarginloss_reference,
    'MultiMarginLoss': multimarginloss_reference,
    'CosineEmbeddingLoss': cosineembeddingloss_reference,
    'TripletMarginLoss': tripletmarginloss_reference,
    'MarginRankingLoss': marginrankingloss_reference,
    'CTCLoss': ctcloss_reference,
}


criterion_tests = [
    dict(
        module_name='L1Loss',
        cpp_constructor_args='',
        input_size=(2, 3, 4),
        target_size=(2, 3, 4),
        reference_fn=lambda i, t, _: 1. / i.numel() *
        sum((a - b).abs().sum() for a, b in zip(i, t)),
    ),
    dict(
        module_name='NLLLoss',
        input_fn=lambda: torch.rand(15, 10).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, None, 2),
        input_fn=lambda: torch.rand(15, 10).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, _: nllloss_reference(i, t, ignore_index=2),
        desc='ignore_index'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m)),
        desc='weights',
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10), None, 2),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m), ignore_index=2),
        desc='weights_ignore_index'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(10), None, -1),
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10 + 1).floor().long() - 1,
        reference_fn=lambda i, t, m:
            nllloss_reference(i, t, weight=get_weight(m), ignore_index=-1),
        desc='weights_ignore_index_neg'
    ),
    dict(
        module_name='KLDivLoss',
        input_fn=lambda: torch.rand(10, 10).log(),
        target_fn=lambda: torch.rand(10, 10),
        reference_fn=lambda i, t, m:
            kldivloss_reference(i, t, get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='MSELoss',
        input_size=(2, 3, 4, 5),
        target_size=(2, 3, 4, 5),
        reference_fn=lambda i, t, m: ((i - t).abs().pow(2).sum() / (i.numel()
                                      if get_reduction(m) == 'mean' else 1)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='BCELoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
        reference_fn=lambda i, t, m: -(t * i.log() + (1 - t) * (1 - i).log()).sum() /
            (i.numel() if get_reduction(m) else 1),
        check_gradgrad=False,
    ),
    dict(
        module_name='BCELoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
        reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() /
            (i.numel() if get_reduction(m) else 1),
        desc='weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input_size=(15, 10),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args_fn=lambda: (torch.rand(10),),
        input_size=(15, 10),
        target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='weights',
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        input_size=(10,),
        target_fn=lambda: torch.randn(10).gt(0).double().mul_(2).sub(1),
        reference_fn=lambda i, t, m:
            hingeembeddingloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        input_size=(10,),
        target_fn=lambda: torch.randn(10).gt(0).double().mul_(2).sub(1),
        reference_fn=lambda i, t, m:
            hingeembeddingloss_reference(i, t, margin=0.5, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(10,),
        target_fn=lambda: torch.rand(10).mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            multilabelmarginloss_reference(i, t, reduction=get_reduction(m)),
        desc="1d",
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(10).floor().long(),
        reference_fn=lambda i, t, m:
            multilabelmarginloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(2).floor(),
        reference_fn=lambda i, t, m: -(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()).sum() / i.numel(),
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(10,),
        target_fn=lambda: torch.rand(1).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, reduction=get_reduction(m)),
        desc='1d',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(2,),
        input_fn=lambda: torch.rand(5, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, p=2, reduction=get_reduction(m)),
        desc='p',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(1, 0.5),
        legacy_constructor_args=(1, None, 0.5),
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, margin=0.5, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        constructor_args=(1, 1., torch.rand(10)),
        legacy_constructor_args=(1, torch.rand(10)),
        input_size=(5, 10),
        target_fn=lambda: torch.rand(5).mul(8).floor().long(),
        reference_fn=lambda i, t, m:
            multimarginloss_reference(i, t, weight=get_weight(m), reduction=get_reduction(m)),
        desc='weights',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(5, 10),
        target_size=(5, 10),
        check_sum_reduction=True,
        reference_fn=lambda i, t, m:
            smoothl1loss_reference(i, t, reduction=get_reduction(m)),
    ),
    dict(
        module_name='SoftMarginLoss',
        input_size=(5, 5),
        target_fn=lambda: torch.randn(5, 5).sign(),
        reference_fn=lambda i, t, m:
            softmarginloss_reference(i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)),
        target_fn=lambda: torch.randn(15).sign(),
        reference_fn=lambda i, t, m:
            cosineembeddingloss_reference(i[0], i[1], t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        constructor_args=(0.7,),
        input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)),
        target_fn=lambda: torch.randn(15).sign(),
        reference_fn=lambda i, t, m:
            cosineembeddingloss_reference(i[0], i[1], t, margin=0.7, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='MarginRankingLoss',
        input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target_fn=lambda: torch.randn(50).sign(),
        reference_fn=lambda i, t, m:
            marginrankingloss_reference(i[0], i[1], t, reduction=get_reduction(m)),
        check_sum_reduction=True,
    ),
    dict(
        module_name='MarginRankingLoss',
        constructor_args=(0.5,),
        input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target_fn=lambda: torch.randn(50).sign(),
        reference_fn=lambda i, t, m:
            marginrankingloss_reference(i[0], i[1], t, margin=0.5, reduction=get_reduction(m)),
        desc='margin',
        check_sum_reduction=True,
    ),
]

new_criterion_tests = [
    dict(
        module_name='BCEWithLogitsLoss',
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        constructor_args=(torch.rand(10),),
        input_fn=lambda: torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(15, 10).gt(0).double(),
        desc='weights',
    ),
    dict(
        module_name='BCEWithLogitsLoss',
        constructor_args=(torch.rand(()),),
        input_fn=lambda: torch.rand(()).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.randn(()).gt(0).double(),
        desc='scalar_weights'
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='2d',
    ),
    dict(
        module_name='NLLLoss',
        constructor_args_fn=lambda: (torch.rand(3),),
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, weight=get_weight(m)),
        desc='2d_weights',
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, None, 1),
        input_size=(2, 3, 5, 5),
        target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, ignore_index=1),
        desc='2d_ignore_index',
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5, 5, 2, 2),
        target_fn=lambda: torch.rand(2, 5, 5, 2, 2).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='higher_dim',
    ),
    dict(
        module_name='NLLLoss',
        input_size=(2, 3, 5),
        target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(),
        reference_fn=lambda i, t, m:
            loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)),
        check_sum_reduction=True,
        desc='dim_is_3',
    ),
    dict(
        module_name='PoissonNLLLoss',  # Default is log_input=True, full=False
        input_size=(2, 3, 4, 5),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _: (i.exp() - t.mul(i)).mean(),
        desc='no_full_loss',
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(False, False),  # log_input=False, full=False
        input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _: (i - t.mul((i + 1e-8).log())).mean(),
        desc='no_full_loss_no_log_input',
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(True, True),  # log_input=True, full=True
        input_size=(2, 3, 4, 5),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _:
            (i.exp() - t.mul(i) + (t.mul(t.log()) - t + 0.5 * (2. * pi * t).log()).masked_fill(t <= 1, 0)).mean(),
        desc='full_loss',
    ),
    dict(
        module_name='PoissonNLLLoss',
        constructor_args=(False, True),  # log_input=False, full=True
        input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001),
        target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(),
        reference_fn=lambda i, t, _:
            (i - t.mul((i + 1e-8).log()) + (t.mul(t.log()) - t + 0.5 * (2. * pi * t).log()).masked_fill(t <= 1, 0)).mean(),
        desc='full_loss_no_log_input',
    ),
    dict(
        module_name='L1Loss',
        cpp_constructor_args='',
        input_size=(),
        target_size=(),
        reference_fn=lambda i, t, _: 1. / i.numel() * (i - t).abs().sum(),
        desc='scalar',
    ),
    dict(
        module_name='KLDivLoss',
        input_fn=lambda: torch.rand(()).log(),
        target_fn=lambda: torch.rand(()),
        reference_fn=lambda i, t, m:
            kldivloss_reference(i, t, get_reduction(m)),
        check_sum_reduction=True,
        desc='scalar',
    ),
    dict(
        module_name='MSELoss',
        input_size=(),
        target_size=(),
        reference_fn=lambda i, t, m: ((i - t).abs().pow(2).sum() /
                                      (i.numel() if get_reduction(m) == 'mean' else 1)),
        check_sum_reduction=True,
        desc='scalar'
    ),
    dict(
        module_name='MSELoss',
        input_fn=lambda: torch.ones(5, 68, 64, 64, dtype=torch.float) / 10,
        target_fn=lambda: torch.zeros(5, 68, 64, 64, dtype=torch.float),
        reference_fn=lambda i, t, m: ((i - t).abs().pow(2).sum() /
                                      (i.numel() if get_reduction(m) == 'mean' else 1)),
        check_forward_only=True,
        desc='prec',
    ),
    dict(
        module_name='BCELoss',
        constructor_args_fn=lambda: (torch.rand(()),),
        input_fn=lambda: torch.rand(()).clamp_(1e-2, 1 - 1e-2),
        target_fn=lambda: torch.rand(()).gt(0).double(),
        reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() /
            (i.numel() if get_reduction(m) == 'mean' else 1),
        desc='scalar_weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        input_size=(),
        target_fn=lambda: torch.randn(()).gt(0).double().mul_(2).sub(1),
        desc='scalar_margin',
        check_sum_reduction=True,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(),
        target_size=(),
        check_sum_reduction=True,
        reference_fn=lambda i, t, m:
            smoothl1loss_reference(i, t, reduction=get_reduction(m)),
        desc='scalar',
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        constructor_args=(torch.rand(10),),
        input_fn=lambda: torch.randn(5, 10),
        target_fn=lambda: torch.rand(5, 10).mul(2).floor(),
        reference_fn=lambda i, t, m: -((t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * get_weight(m)).sum() /
            (i.numel() if get_reduction(m) == 'mean' else i.size(1) if get_reduction(m) == 'sum' else 1),
        desc='weights',
        check_sum_reduction=True,
        check_gradgrad=False,
    ),
    dict(
        module_name='CTCLoss',
        constructor_args=(14,),  # blank=14
        extra_args=([50, 50, 50], [30, 25, 20]),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
    ),
    dict(
        module_name='CTCLoss',
        desc='1d_target',
        constructor_args=(14,),  # blank=14
        extra_args=([50, 50, 50], [30, 25, 20]),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
    ),
    dict(
        module_name='CTCLoss',
        desc='2d_int_target',
        constructor_args=(0,),  # blank=0
        extra_args=([50, 50, 50], [30, 25, 20]),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
        convert_target=False,
    ),
    dict(
        module_name='CTCLoss',
        desc='2d_lengths_tensors',
        constructor_args=(0,),  # blank=0
        extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])),  # input_lengths, target_lengths
        input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2),
        target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int),
        reference_fn=lambda i, t, il, tl, m:
            ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)),
        check_sum_reduction=True,
        check_gradgrad=False,
        check_half=False,
        convert_target=False,
    ),
]


class NNTestCase(TestCase):

    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)

    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        else:
            return tuple(self._flatten_tensors(a) for a in x)

    def _zero_grad_input(self, input):
        if isinstance(input, torch.Tensor):
            if input.requires_grad and input.grad is not None:
                input.grad.zero_()
                input.grad.detach_()
        else:
            for i in input:
                self._zero_grad_input(i)

    def _analytical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nelement()

        if jacobian_input:
            jacobian_inp = self._jacobian(input, output_size)
            flat_jacobian_input = list(iter_tensors(jacobian_inp))

        if jacobian_parameters:
            num_param = sum(p.numel() for p in self._get_parameters(module)[0])
            jacobian_param = torch.zeros(num_param, output_size)

        for i in range(output_size):
            param, d_param = self._get_parameters(module)
            # make non grad zeros
            d_param = [torch.zeros_like(p) if d is None else d for (p, d) in zip(param, d_param)]

            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1

            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # Tensors will accumulate gradient from multiple steps
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)

            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x.contiguous().view(-1)
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)

        res = tuple()
        if jacobian_input:
            res += jacobian_inp,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    def _numerical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        def fw(input):
            return self._forward(module, input).detach()

        res = tuple()
        if jacobian_input:
            res += get_numerical_jacobian(fw, input, eps=1e-6),
        if jacobian_parameters:
            param, _ = self._get_parameters(module)
            res += torch.cat([get_numerical_jacobian(fw, input, p, eps=1e-6) for p in param], 0),
        return res

    def check_jacobian(self, module, input, jacobian_input=True):
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = list(iter_tensors(analytical))
        numerical_t = list(iter_tensors(numerical))

        # TODO: compare structure
        if input.numel() != 0:
            self.assertLessEqual(
                max(a.add(-1, n).abs().max() for a, n in zip(analytical_t, numerical_t)),
                PRECISION
            )

    def check_criterion_jacobian(self, criterion, input, target):
        eps = 1e-6
        self._forward_criterion(criterion, input, target)
        analytical_d_x = self._backward_criterion(criterion, input, target)
        numerical_d_x = deepcopy(analytical_d_x)

        input_t = iter_tensors(input)
        numerical_t = iter_tensors(numerical_d_x)
        for x, d_x in zip(input_t, numerical_t):
            x = x.view(-1).data
            d_x = d_x.view(-1).data
            for i in range(x.nelement()):
                original = x[i].item()
                x[i] = original + eps
                fx1 = self._forward_criterion(criterion, input, target)
                x[i] = original - eps
                fx2 = self._forward_criterion(criterion, input, target)
                deriv = (fx1 - fx2) / (2. * eps)
                d_x[i] = float(deriv)
                x[i] = original

        # TODO: check structure
        analytical_t = list(iter_tensors(analytical_d_x))
        numerical_t = list(iter_tensors(numerical_d_x))

        self.assertLessEqual(
            max(a.add(-1, n).abs().max() for a, n in zip(analytical_t, numerical_t)),
            PRECISION
        )


class TestBase(object):

    _required_arg_names = {'constructor_args', 'input', 'extra_args'}

    def __init__(self, constructor, desc='', reference_fn=None, fullname=None, **kwargs):
        self.desc = desc
        self.fullname = fullname
        self.constructor = constructor
        self.reference_fn = reference_fn
        for name in self._required_arg_names:
            if name not in kwargs and name + '_fn' not in kwargs and name + '_size' not in kwargs:
                if name in {'constructor_args', 'extra_args'}:
                    kwargs[name] = tuple()
                else:
                    raise ValueError("{}: Specify {} by a value, a function to generate it, or it's size!"
                                     .format(self.get_name(), name))
        self._extra_kwargs = kwargs
        self._arg_cache = {}

    def get_name(self):
        if self.fullname is not None:
            return 'test_' + self.fullname

        test_name = 'test_' + self.constructor.__name__
        if self.desc:
            test_name += '_' + self.desc
        return test_name

    def _unpack(self, value):
        if isinstance(value, torch.Tensor):
            return value
        elif is_iterable(value):
            return type(value)(self._unpack(v) for v in value)
        else:
            return value

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', True)

    @property
    def extra_args(self):
        return self._get_arg('extra_args', True)

    def _get_arg(self, name, unpack):
        assert name in self._required_arg_names

        if name not in self._arg_cache:
            fn_name = name + '_fn'
            size_name = name + '_size'

            if name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[name]
            elif fn_name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[fn_name]()
            else:
                assert size_name in self._extra_kwargs, \
                    "Missing `{}`, `{}` or `{}` for {}".format(name, size_name, fn_name, self.get_name())

                def map_tensor_sizes(sizes):
                    if isinstance(sizes, list):
                        return [map_tensor_sizes(s) for s in sizes]
                    elif isinstance(sizes, torch.Tensor):
                        return sizes.double()
                    else:
                        return torch.randn(sizes)

                self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])

        return self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name]

    def _get_input(self, unpack=True):
        return self._get_arg('input', unpack)

    def __call__(self, test_case):
        raise NotImplementedError


class ModuleTest(TestBase):

    def __init__(self, *args, **kwargs):
        super(ModuleTest, self).__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.should_test_pickle = kwargs.get('pickle', True)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.FIXME_no_cuda_gradgrad_comparison = \
            kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        self.precision = kwargs.get('precision', 2e-4)
        self.check_forward_only = kwargs.get('check_forward_only', False)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        if self.reference_fn is not None:
            out = test_case._forward(module, input)
            ref_input = deepcopy(input)
            ref_module = deepcopy(module)
            expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0], ref_module)
            test_case.assertEqual(out, expected_out)
        if self.check_forward_only:
            return
        self.test_noncontig(test_case, module, input)

        if self.should_test_pickle:
            # TODO: do this with in-memory files as soon as torch.save will support it
            with TemporaryFile() as f:
                test_case._forward(module, input)
                torch.save(module, f)
                f.seek(0)
                module_copy = torch.load(f)
                test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))

        self._do_test(test_case, module, input)

    def noncontiguize(self, obj):
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        tensor = obj
        ndim = tensor.dim()
        # Always making only the last dimension noncontiguous is easy to hide
        # bugs because .view(-1) will still work. So try to find a dim with size
        # > 1 and make that non-contiguous, i.e., stack + select on the
        # dimension directly after that.
        dim = ndim
        for d in range(ndim):
            if tensor.size(d) > 1:
                dim = d + 1
                break
        noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
        assert noncontig.numel() == 1 or noncontig.numel() == 0 or not noncontig.is_contiguous()
        noncontig.requires_grad = tensor.requires_grad
        return noncontig

    def test_noncontig(self, test_case, module, input):
        # check no scalars, can't make non-contig
        if isinstance(input, torch.Tensor) and input.dim() == 0:
            return
        if any(i.dim() == 0 for i in input if isinstance(i, torch.Tensor)):
            return

        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            grad_output = output.new(output.shape).normal_()
            output = output.clone()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])

        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for contig_i, contig_g in product((True, False), repeat=2):
            i = input if contig_i else nc_input
            # Some ops, e.g., nn.Flatten, return gradient that shares
            # storage with the grad_output. Hence we copy here.
            go = deepcopy(grad_output if contig_g else nc_grad_output)
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                out = test_case._forward(module, i)
                grad = test_case._backward(module, i, out, go)

                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, 1e-4)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {'torch.DoubleTensor': torch.cuda.FloatTensor}
            gpu_input = to_gpu(cpu_input, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0]):
                gpu_p.data.copy_(cpu_p)

            test_case._zero_grad_input(cpu_input)
            test_case._zero_grad_input(gpu_input)
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_output = test_case._forward(cpu_module, cpu_input)
            gpu_output = test_case._forward(gpu_module, gpu_input)
            test_case.assertEqual(cpu_output, gpu_output, self.precision)

            # Run backwards on CPU and GPU and compare results
            for _ in range(5):
                cpu_gradOutput = cpu_output.clone().normal_()
                gpu_gradOutput = cpu_gradOutput.type('torch.cuda.FloatTensor')
                cpu_gradInput = test_case._backward(cpu_module, cpu_input, cpu_output, cpu_gradOutput)
                gpu_gradInput = test_case._backward(gpu_module, gpu_input, gpu_output, gpu_gradOutput)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, self.precision)
                for cpu_d_p, gpu_d_p in zip(cpu_param[1], gpu_param[1]):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, self.precision)

            # Run double-backwards on CPU and GPU and compare results
            if self.check_gradgrad and not self.FIXME_no_cuda_gradgrad_comparison:
                cpu_output = cpu_module(cpu_input)
                gpu_output = gpu_module(gpu_input)

                cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output).detach()
                gpu_gradOutput.requires_grad = True

                cpu_gradInputs = torch.autograd.grad(
                    cpu_output,
                    (cpu_input,) + tuple(cpu_module.parameters()),
                    cpu_gradOutput,
                    create_graph=True)
                gpu_gradInputs = torch.autograd.grad(
                    gpu_output,
                    (gpu_input,) + tuple(gpu_module.parameters()),
                    gpu_gradOutput,
                    create_graph=True)

                for cpu_d_i, gpu_d_i in zip(cpu_gradInputs, gpu_gradInputs):
                    test_case.assertEqual(cpu_d_i, gpu_d_i, self.precision)

                # We mix output into the second backwards computation so that
                # torch.autograd.grad doesn't complain that some inputs
                # are unreachable (which can happen if you differentiate
                # only on the gradient.
                cpu_gg = torch.autograd.grad(
                    cpu_output.sum() + sum(map(lambda x: x.sum(), cpu_gradInputs)),
                    (cpu_input, cpu_gradOutput) + tuple(cpu_module.parameters()),
                    retain_graph=True)
                gpu_gg = torch.autograd.grad(
                    gpu_output.sum() + sum(map(lambda x: x.sum(), gpu_gradInputs)),
                    (gpu_input, gpu_gradOutput) + tuple(gpu_module.parameters()),
                    retain_graph=True)

                test_case.assertEqual(cpu_gradInput, gpu_gradInput, self.precision)
                for cpu_d_p, gpu_d_p in zip(cpu_gg, gpu_gg):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, self.precision)

            self.test_noncontig(test_case, gpu_module, gpu_input)
        except NotImplementedError:
            pass
        # TODO: remove this after CUDA scatter_ is implemented
        except AttributeError as e:
            if len(e.args) == 1 and "'FloatTensor' object has no attribute 'scatter_'" in e.args[0]:
                pass
            else:
                raise


class CriterionTest(TestBase):

    _required_arg_names = TestBase._required_arg_names.union({'target'})

    def __init__(self, *args, **kwargs):
        super(CriterionTest, self).__init__(*args, **kwargs)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.check_forward_only = kwargs.get('check_forward_only', True)

    def _get_target(self):
        return self._get_arg('target', True)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        # Check that these methods don't raise errors
        module.__repr__()
        str(module)

        target = self._get_target()

        if self.reference_fn is not None:
            out = test_case._forward_criterion(module, input, target, extra_args=self.extra_args)
            ref_args = (deepcopy(input), deepcopy(target)) + self.extra_args + (module,)
            expected_out = self.reference_fn(*ref_args)
            test_case.assertEqual(out, expected_out)

        if self.check_forward_only:
            return

        test_case.check_criterion_jacobian(module, input, target)
        self._do_extra_tests(test_case, module, input, target)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {
                'torch.DoubleTensor': torch.cuda.FloatTensor,
            }
            gpu_input = to_gpu(cpu_input, type_map=type_map)

            cpu_target = self._get_target()
            gpu_target = to_gpu(cpu_target, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_output, gpu_output, 4e-4)

            gradOutput = torch.randn(())
            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target, gradOutput)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target, gradOutput)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 4e-4)
        except NotImplementedError:
            pass

    def _do_extra_tests(self, test_case, module, input, target):
        pass
