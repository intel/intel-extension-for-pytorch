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

def _assertGradAndGradgradChecks(test_case, apply_fn, inputs):
    # call assert function rather than returning a bool since it's nicer
    # if we get whether this failed on the gradcheck or the gradgradcheck.
    test_case.assertTrue(gradcheck(apply_fn, inputs))
    test_case.assertTrue(gradgradcheck(apply_fn, inputs))

device = ipex.DEVICE
#device = 'cpu:0'
SIZE = 100

class TestOP(TestCase):
    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True):
        float_types = [torch.double,
                       torch.float]
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        def make_contiguous(shape, dtype):
            if dtype in float_types:
                val = torch.randn(shape, dtype=dtype, device=device)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            result = torch.randint(val_range[0], val_range[1] + 1, shape, dtype=dtype, device=device)
            return result

        def make_non_contiguous(shape, dtype):
            contig = make_contiguous(shape, dtype)
            non_contig = torch.empty(shape + (2, 2), dtype=dtype, device=device)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        def make_contiguous_slice(size, dtype):
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            self.assertTrue(non_contig.is_contiguous())
            return contig

        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        tensors = {"cont": [], "noncont": [], "slice": []}
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(make_contiguous_slice(sum(list(shape)), dtype))

        return tensors

    def _test_math(self, torchfn, mathfn, input=None, test_expand=False):
        if input is None:
            input = []
            input.append(list(range(-5, 5)))
            input.append([0 for x in range(-5, 5)])
            input.append([x + 1e-6 for x in range(-5, 5)])
            # Some vectorized implementations don't support large ranges
            input.append([x + 1e10 for x in range(-5, 5)])
            input.append([x - 1e10 for x in range(-5, 5)])
            input.append(torch.randn(10).tolist())
            input.append((torch.randn(10) + 1e6).tolist())
            input.append([math.pi * (x / 2) for x in range(-5, 5)])

        def compare_reference(input, dtype):
            input = torch.tensor(input, dtype=dtype, device=device)
            res1 = torchfn(input.clone())
            res2 = input.clone()
            res3 = res2.view(res2.numel())
            for i in range(0, res3.size(0)):
                res3[i] = mathfn(res3[i])
            torch.testing.assert_allclose(res1, res2)

        # compare against the reference math function
        compare_reference(input, torch.double)
        compare_reference(input, torch.float)

        def check_non_contiguous(shape, dtype):
            contig = torch.randn(shape, dtype=dtype, device=device)
            non_contig = torch.empty(shape + (2,), dtype=dtype, device=device)[..., 0]
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(non_contig), 'non-contiguous')

        # compare application against contiguous vs. non-contiguous
        check_non_contiguous((5, 7), torch.double)
        check_non_contiguous((1024,), torch.double)
        check_non_contiguous((5, 7), torch.float)
        check_non_contiguous((1024,), torch.float)

        def check_non_contiguous_index(dtype):
            contig = torch.randn((2, 2, 1, 2), dtype=dtype, device=device)
            non_contig = contig[:, 1, ...]
            contig = non_contig.clone()
            self.assertFalse(non_contig.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(non_contig), 'non-contiguous index')

        check_non_contiguous_index(torch.float)
        check_non_contiguous_index(torch.double)

        def check_non_contiguous_expand(shape, dtype):
            contig = torch.randn(shape, dtype=dtype, device=device)
            non_contig = contig.clone().expand(3, -1, -1)
            self.assertFalse(non_contig.is_contiguous())
            contig = torchfn(contig)
            non_contig = torchfn(non_contig)
            for i in range(3):
                self.assertEqual(contig, non_contig[i], 'non-contiguous expand[' + str(i) + ']')

        # Expand is not defined for in-place operations
        if test_expand:
            # The size 1 case is special as it leads to 0 stride and needs to persists
            check_non_contiguous_expand((1, 3), torch.double)
            check_non_contiguous_expand((1, 7), torch.double)
            check_non_contiguous_expand((5, 7), torch.float)

        # If size(dim) == 1, stride(dim) is not defined.
        # The code needs to be able to handle this
        def check_contiguous_size1(dtype):
            contig = torch.randn((5, 100), dtype=dtype, device=device)
            contig = contig[:1, :50]
            contig2 = torch.empty(contig.size(), dtype=dtype, device=device)
            contig2.copy_(contig)
            self.assertTrue(contig.is_contiguous())
            self.assertTrue(contig2.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(contig2), 'contiguous size1')

        check_contiguous_size1(torch.double)
        check_contiguous_size1(torch.float)

        def check_contiguous_size1_largedim(dtype):
            contig = torch.randn((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), dtype=dtype, device=device)
            contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
            contig2 = torch.empty(contig.size(), dtype=dtype, device=device)
            contig2.copy_(contig)
            self.assertTrue(contig.is_contiguous())
            self.assertTrue(contig2.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(contig2), 'contiguous size1')

        check_contiguous_size1_largedim(torch.double)
        check_contiguous_size1_largedim(torch.float)

        def check_large(dtype):
            input = torch.randn(1024, 512, dtype=dtype, device=device)
            actual = torchfn(input)
            expected = torch.stack([torchfn(slice) for slice in input])
            self.assertEqual(actual, expected, 'large')

        # compare large tensor vs. repeated small applications to expose
        # possible parallelism bugs.
        check_large(torch.double)
        check_large(torch.float)

    def __test_math_by_name(self, function_name, mathfn, selffn):
        mathfn = getattr(math, mathfn)
        if selffn:
            def torchfn(x):
                return getattr(x, function_name)()
        else:
            torchfn = getattr(torch, function_name)
        self._test_math(torchfn, mathfn, test_expand=(not selffn))

    def _test_math_by_name(self, function_name, test_self=True):
        if test_self:
            self.__test_math_by_name(function_name + "_", function_name, True)
        self.__test_math_by_name(function_name, function_name, False)

    def test_ceil(self):
        self._test_math_by_name('ceil')

    def test_ones(self):
        res1 = torch.ones(3, device=device)
        mask = torch.tensor([True, False, True], device=device)
        res1[mask] = 3
        res2 = torch.tensor([3.0, 1.0, 3.0], device=device)
        self.assertEqual(res1, res2)

    def test_add(self):
        m1 = torch.randn(100, 100, device=device)
        v1 = torch.randn(100, device=device)

        # contiguous
        res1 = torch.add(m1[1], v1)
        res2 = torch.zeros(100, device=device)
        for i in range(m1.size(1)):
            res2[i] = m1[1, i] + v1[i]
        self.assertEqual(res1, res2)

        # non-contiguous
        res1 = torch.add(m1[:, 4], v1)
        res2 = res1.clone().zero_()
        for i in range(m1.size(0)):
            res2[i] = m1[i, 4] + v1[i]
        self.assertEqual(res1, res2)

        # [res] torch.add([res,] tensor, value)
        m1 = torch.randn(10, 10, device=device)

        # contiguous
        res1 = m1.clone()
        res1[3].add_(2)
        res2 = m1.clone()
        for i in range(m1.size(1)):
            res2[3, i] = res2[3, i] + 2
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10, device=device)
        res1 = m1.clone()
        res1[:, 3].add_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] + 2
        self.assertEqual(res1, res2)

        # contiguous + non-contiguous
        m1 = torch.randn(10, 10, device=device)
        m2 = torch.randn(10, 10, device=device).t()
        res = m1 + m2
        self.assertTrue(res.is_contiguous())
        self.assertEqual(res, m1 + m2.contiguous())

        inputs1_cpu = torch.randn(100, 100, requires_grad=True)
        inputs2_cpu = torch.randn(100, 100, requires_grad=True)
        inputs1_dpcpp = inputs1_cpu.detach().to(device=device).requires_grad_(True)
        inputs2_dpcpp = inputs2_cpu.detach().to(device=device).requires_grad_(True)
        out_dpcpp = torch.add(inputs1_dpcpp, inputs2_dpcpp)
        out_cpu = torch.add(inputs1_cpu, inputs2_cpu)
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs1_dpcpp.grad.to('cpu'), inputs1_cpu.grad, prec=1e-4)
        self.assertEqual(inputs2_dpcpp.grad.to('cpu'), inputs2_cpu.grad, prec=1e-4)

    def test_sub(self):
        m1 = torch.tensor([2.34, 4.44], dtype=torch.float32, device=device)
        m2 = torch.tensor([1.23, 2.33], dtype=torch.float32, device=device)
        self.assertEqual(m1 - m2, torch.tensor([1.11, 2.11], dtype=torch.float32, device=device))

    def test_mul(self):
        a1 = torch.tensor([True, False, False, True], dtype=torch.bool, device=device)
        a2 = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
        self.assertEqual(a1 * a2, torch.tensor([True, False, False, False], dtype=torch.bool, device=device))

        a1 = torch.tensor([0.1, 0.1], device=device)
        a2 = torch.tensor([1.1, 0.1], device=device)
        self.assertEqual(a1 * a2, torch.tensor([0.11, 0.01], device=device))
        self.assertEqual(a1.mul(a2), a1 * a2)

    def test_div(self):
        a1 = torch.tensor([4.2, 6.2], device=device)
        a2 = torch.tensor([2., 2.], device=device)
        self.assertEqual(a1 / a2, torch.tensor([2.1, 3.1], device=device))
        self.assertEqual(a1.div(a2), a1 / a2)

        m1 = torch.randn(10, 10, dtype=torch.float, device=device)
        res1 = m1.clone()
        res1[:, 3].div_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] / 2
        self.assertEqual(res1, res2)

    def test_topk(self):
        def topKViaSort(t, k, dim, dir):
            sorted, indices = t.sort(dim, dir)
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # Values should be exactly equivalent
            self.assertEqual(res1, res2, 0)

            # Indices might differ based on the implementation, since there is
            # no guarantee of the relative order of selection
            if not ind1.eq(ind2).all():
                # To verify that the indices represent equivalent elements,
                # gather from the input using the topk indices and compare against
                # the sort indices
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, 0)

        def compare(t, k, dim, dir):
            topKVal, topKInd = t.topk(k, dim, dir, True)
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        t = torch.rand(random.randint(1, SIZE),
                       random.randint(1, SIZE),
                       random.randint(1, SIZE))

        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

    def test_view(self):
        tensor = torch.rand(15, device=device)
        template = torch.rand(3, 5, device=device)
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)

        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        empty = torch.empty(0, device=device)
        self.assertEqual(empty.view_as(empty), empty)
        self.assertEqual(empty.view(0), empty)
        self.assertEqual(empty.view(0, 3, 0, 1).size(), torch.Size([0, 3, 0, 1]))
        self.assertEqual(empty.view(0, 3, 0, 1).view(0), empty)

        # test size inference with empty tensors
        self.assertEqual(empty.view(-1).size(), torch.Size([0]))
        self.assertEqual(empty.view(10, 3, -1).size(), torch.Size([10, 3, 0]))

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(-1, 0)

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(3, 0, -1, 0)

        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))

        # TODO(Eikan): DNNL OP does not support >6 dim tensor, so we disable it temporily. When we fix it, we will open it
        old_dnnl_conf = ipex.core.get_auto_dnnl()
        ipex.core.disable_auto_dnnl()
        # test view when tensor is not contiguous in every dimension, but only
        # contiguous dimensions are touched.
        tensor = torch.rand(4, 2, 5, 1, 6, 2, 9, 3, device=device).transpose(-1, 2).transpose(-2, 3)
        # size:                      [   4,    2,    3,    9,    6,    2,    1,    5]
        # stride:                    [3840, 1620,    1,    3,   54,   27,  324,  324]
        # contiguous dim chunks:     [__________, ____, ____, __________, ____, ____]
        # merging 1 to chunk after:  [__________, ____, ____, __________, __________]
        contig_tensor = tensor.clone()
        # [4, 2] => [8, 1]
        # [3] => [3]
        # [9] => [3, 3]
        # [6, 2] => [4, 1, 3]
        # [1, 5] => [5]
        view_size = [8, 1, 3, 3, 3, 4, 1, 3, 5]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        # [4, 2] => [2, 4]
        # [3] => [3]
        # [9] => [1, 9]
        # [6, 2] => [2, 2, 3]
        # [1, 5] => [5, 1]
        view_size = [2, 4, 3, 1, 9, 2, 2, 3, 5, 1]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        # adding size 1 dims
        view_size = [1, 1, 2, 1, 4, 3, 1, 1, 9, 1, 2, 1, 2, 3, 1, 5, 1, 1]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        if old_dnnl_conf:
            ipex.core.enable_auto_dnnl()
        else:
            ipex.core.disable_auto_dnnl()

        # invalid views
        self.assertRaises(RuntimeError, lambda: tensor.view(-1))
        # crossing [4, 2], [3]
        self.assertRaises(RuntimeError, lambda: tensor.view(24, 9, 6, 2, 1, 5))
        # crossing [6, 2], [1, 5]
        self.assertRaises(RuntimeError, lambda: tensor.view(8, 3, 9, 6, 10))
        # crossing [9], [6, 2]
        self.assertRaises(RuntimeError, lambda: tensor.view(8, 3, 54, 2, 1, 5))

        # view with stride 0 dims
        tensor = torch.empty(1, 1, device=device).expand(3, 4)  # all dims are contiguous
        contig_tensor = tensor.clone()
        self.assertEqual(tensor.view(-1), contig_tensor.view(-1))
        self.assertEqual(tensor.view(1, -1, 1), contig_tensor.view(1, -1, 1))
        self.assertEqual(tensor.view(-1, 1), contig_tensor.view(-1, 1))
        self.assertEqual(tensor.view(6, 2, 1), contig_tensor.view(6, 2, 1))
        self.assertEqual(tensor.view(1, 6, 2, 1), contig_tensor.view(1, 6, 2, 1))

        inputs_cpu = torch.randn(0, 6, requires_grad=True)
        inputs_dpcpp = inputs_cpu.detach().to(device=device).requires_grad_(True)
        out_dpcpp = inputs_dpcpp.view(1, 0, 6, 1, 1)
        out_cpu = inputs_cpu.view(1, 0, 6, 1, 1)
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs_dpcpp.grad.to('cpu'), inputs_cpu.grad, prec=1e-4)

    def test_abs(self):
        def _test_abs(tensors_dict):
            for _category, tensors in tensors_dict.items():
                for data in tensors:
                    _test_abs_single(data)

        def _test_abs_single(data):
            switch = torch.rand(data.size(), device=device).mul(2).floor().mul(2).add(-1).type(data.dtype)
            res = torch.mul(data, switch)
            self.assertTensorsSlowEqual(res.abs(), data, 1e-16)

        shapes = [(3, 4), (3, 5, 7), (2, 2, 5, 8, 2, 3), (1000,), (10, 10, 10)]

        for shape in shapes:
            # Test all except char/byte
            _test_abs(self._make_tensors(shape, val_range=(0, 1000)))

            # Test char
            _test_abs_single(torch.CharTensor(*shape).random_(0, 100))

            # Test byte
            byte_tensor = torch.ByteTensor(*shape).random_(0, 100)
            self.assertTensorsSlowEqual(byte_tensor, byte_tensor.abs(), 1e-16)

        # Checking that the right abs function is called for LongTensor
        bignumber = 2 ** 31 + 1
        res = torch.LongTensor((-bignumber,))
        self.assertGreater(res.abs()[0], 0)

        # One of
        rec = torch.randn(2, 2, 3, 7, 6, 2).type(torch.float64).clamp(0, 1)
        val1 = rec.select(-1, -1).data[0][0][0].sum()
        val2 = rec.select(-1, -1).data.abs()[0][0][0].sum()
        self.assertEqual(val1, val2, 1e-8, 'absolute value')

        # Both abs(0.0) and abs(-0.0) should result in 0.0
        for dtype in (torch.float, torch.double):
            for abs_zeros in (torch.tensor([0.0, -0.0], dtype=dtype).abs().tolist(),
                              # test a large tensor so that the vectorized version is tested
                              torch.abs(-torch.zeros(10000, dtype=dtype)).tolist()):
                for num in abs_zeros:
                    self.assertGreater(math.copysign(1.0, num), 0.0)

    def test_resize(self):
        x = torch.ones(2, 3, device=device)
        self.assertTrue(x.resize(3, 2).size() == (3, 2))

    def test_cat(self):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, 0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, 0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, 0)

        x = torch.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE, device=device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    def test_log_softmax(self):
        x_small = torch.ones(1, 2, device=device)
        x_big = x_small + 1e16
        self.assertEqual(F.log_softmax(x_small, -1), F.log_softmax(x_big, -1))

        inputf = torch.rand(32, 100, device=device, dtype=torch.float, requires_grad=True)
        outf = F.log_softmax(inputf, dim=-1)
        outf.sum().backward()
        self.assertEqual(inputf.grad.dtype, torch.float)

        inputs_cpu = torch.rand(32, 100, requires_grad=True)
        inputs_dpcpp = inputs_cpu.detach().to(device=device).requires_grad_(True)
        out_dpcpp = F.log_softmax(inputs_dpcpp, dim=-1)
        out_cpu = F.log_softmax(inputs_cpu, dim=-1)
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs_dpcpp.grad.to('cpu'), inputs_cpu.grad, prec=1e-4)

    def test_nll_loss_mismatched_batch(self):
        x = torch.randn((10, 3), device=device, requires_grad=True)
        # t should have size (10,)
        t = torch.zeros((3,), device=device, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

        inputs_cpu = torch.randn((10, 3), requires_grad=True)
        inputs_dpcpp = inputs_cpu.detach().to(device=device).requires_grad_(True)
        t_cpu = torch.zeros((10,), dtype=torch.int64)
        t_dpcpp = t_cpu.to(device=device)
        out_dpcpp = F.nll_loss(inputs_dpcpp, t_dpcpp)
        out_cpu = F.nll_loss(inputs_cpu, t_cpu)
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs_dpcpp.grad.to('cpu'), inputs_cpu.grad, prec=1e-4)

    # def test_nll_loss_out_of_bounds_ignore_index(self):
    #     x = torch.randn(6, 3, requires_grad=True)
    #     t = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64)
    #     for reduction in ['mean', 'none']:
    #         F.nll_loss(x, t, ignore_index=255, reduction=reduction).sum().backward()

    def test_poisson_nll_loss_reduction_modes(self):
        input = torch.tensor([0.5, 1.5, 2.5], device=device)
        target = torch.tensor([1., 2., 3.], device=device)
        component_wise_loss = torch.exp(input) - target * input
        self.assertEqual(component_wise_loss,
                         F.poisson_nll_loss(input, target, reduction='none'))
        self.assertEqual(torch.sum(component_wise_loss),
                         F.poisson_nll_loss(input, target, reduction='sum'))
        self.assertEqual(torch.mean(component_wise_loss),
                         F.poisson_nll_loss(input, target, reduction='mean'))
        with self.assertRaisesRegex(ValueError, 'is not valid'):
            F.poisson_nll_loss(input, target, reduction='total')

    def test_to_cpu(self):
        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True))

            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append('cuda:{}'.format(torch.cuda.current_device()))
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        a = torch.tensor(5, device='cpu:0')
        test_copy_behavior(a)
        self.assertEqual(a.device, a.to('cpu:0').device)
        self.assertEqual(a.device, a.to('cpu', dtype=torch.float32).device)
        self.assertIs(torch.float32, a.to('cpu', dtype=torch.float32).dtype)
        self.assertEqual(a.device, a.to(torch.float32).device)
        self.assertIs(torch.float32, a.to(dtype=torch.float32).dtype)
        self.assertEqual(a.data_ptr(), a.to('cpu').data_ptr())
        self.assertEqual(a.data_ptr(), a.to(dtype=a.dtype, device=a.device, copy=False).data_ptr())
        self.assertEqual(a.data_ptr(), a.to('cpu', copy=False).data_ptr())
        self.assertNotEqual(a.data_ptr(), a.to('cpu', copy=True).data_ptr())

    def test_to(self):
        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True))

            devices = [t.device]
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        a = torch.tensor(5, device='cpu')
        b = a.to(ipex.DEVICE)
        a_clone = a.clone().view(a.numel())
        b_clone = b.clone().view(b.numel())
        self.assertEqual(a.size(), b.size())
        for i in range(0, b.numel()):
            assert (a_clone[i] == b_clone[i])

        a = torch.tensor(5, device=ipex.DEVICE)
        self.assertEqual(a.device, a.to(ipex.DEVICE).device)

    def test_index(self):
        s = [2, 3, 1, 8]
        a = torch.randn(3, 10, device='cpu')
        b = a.to(ipex.DEVICE)
        self.assertEqual(a[:, s].to(ipex.DEVICE), b[:, s])
        self.assertEqual(a[1:, s].to(ipex.DEVICE), b[1:, s])
        self.assertEqual(a[:1, s].to(ipex.DEVICE), b[:1, s])

    def test_addmm(self):
        M_cpu = torch.randn(10, 25, requires_grad=True)
        m1_cpu = torch.randn(10, 50, requires_grad=True)
        m2_cpu = torch.randn(50, 25, requires_grad=True)
        M_dpcpp = M_cpu.detach().to(device=device).requires_grad_(True)
        m1_dpcpp = m1_cpu.detach().to(device=device).requires_grad_(True)
        m2_dpcpp = m2_cpu.detach().to(device=device).requires_grad_(True)
        out_dpcpp = torch.addmm(M_dpcpp, m1_dpcpp, m2_dpcpp)
        out_cpu = torch.addmm(M_cpu, m1_cpu, m2_cpu)
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(M_dpcpp.grad.to('cpu'), M_cpu.grad, prec=1e-4)
        self.assertEqual(m1_dpcpp.grad.to('cpu'), m1_cpu.grad, prec=1e-4)
        self.assertEqual(m2_dpcpp.grad.to('cpu'), m2_cpu.grad, prec=1e-4)

    def test_mean(self):
        inputs_cpu = torch.randn(1, 2, 3, 4, requires_grad=True)
        inputs_dpcpp = inputs_cpu.detach().to(device=device).requires_grad_(True)
        out_dpcpp = inputs_dpcpp.mean()
        out_cpu = inputs_cpu.mean()
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs_dpcpp.grad.to('cpu'), inputs_cpu.grad, prec=1e-4)

    def test_relu(self):
        inputs_cpu = torch.randn(1, 2, 3, 4, requires_grad=True)
        inputs_dpcpp = inputs_cpu.detach().to(device=device).requires_grad_(True)
        out_dpcpp = inputs_dpcpp.relu()
        out_cpu = inputs_cpu.relu()
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)

        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs_dpcpp.grad.to('cpu'), inputs_cpu.grad, prec=1e-4)


class TestBN(TestCase):
    def test_batchnorm_raises_error_if_running_mean_is_not_same_size_as_input(self):
        input = torch.rand(2, 10, device=device)
        running_var = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, torch.rand(size, device=device), running_var)

    def test_batchnorm_raises_error_if_running_var_is_not_same_size_as_input(self):
        input = torch.rand(2, 10, device=device)
        running_mean = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, torch.rand(size, device=device))

    def test_batchnorm_raises_error_if_weight_is_not_same_size_as_input(self):
        input = torch.rand(2, 10, device=device)
        running_mean = torch.rand(10, device=device)
        running_var = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, weight=Parameter(torch.rand(size, device=device)))

    def test_batchnorm_raises_error_if_bias_is_not_same_size_as_input(self):
        input = torch.rand(2, 10, device=device)
        running_mean = torch.rand(10, device=device)
        running_var = torch.rand(10, device=device)
        wrong_sizes = [9, 11]
        for size in wrong_sizes:
            with self.assertRaises(RuntimeError):
                F.batch_norm(input, running_mean, running_var, bias=Parameter(torch.rand(size, device=device)))

    def test_batchnorm_eval(self):
        device = 'cpu:0'
        torch.manual_seed(123)
        input = torch.rand(2, 10, device=device)
        running_var = torch.rand(10, device=device)
        res1 = F.batch_norm(input, torch.rand(10, device=device), running_var)

        device = ipex.DEVICE
        torch.manual_seed(123)
        input = torch.rand(2, 10, device=device)
        running_var = torch.rand(10, device=device)
        res2 = F.batch_norm(input, torch.rand(10, device=device), running_var)

        res1_clone = res1.clone().view(res1.numel())
        res2_clone = res2.clone().view(res2.numel())
        self.assertEqual(res1.size(), res2.size())
        for i in range(0, res2.numel()):
            assert (res1_clone[i] == res2_clone[i])


class TestAvgMaxPool(TestCase):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce((lambda x, y: x * y), kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_doubletensor_avg_pool2d(self):
        n, m = 5, 8
        input = torch.rand(1, 1, n, m, device=device)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                actual = torch.nn.functional.avg_pool2d(input[0], (i, j))
                actual = actual.view(1, actual.numel())
                expected = self._avg_pool2d(input, (i, j))
                self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_avg_pool2d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: torch.nn.functional.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m, device=device)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = torch.nn.functional.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_doubletensor_avg_pool3d(self):
        h, w, d = 5, 6, 7
        input = torch.rand(h, w, d, device=device)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    actual = torch.nn.functional.avg_pool3d(input.unsqueeze(0), (i, j, k))
                    actual = actual.view(1, actual.numel())
                    expected = self._avg_pool3d(input, (i, j, k))
                    self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-5))

    def test_avg_pool3d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: torch.nn.functional.avg_pool3d(torch.zeros(3, 3, 3, 3), (2, 2, 2), divisor_override=0))

    @unittest.expectedFailure
    def test_max_pool_nan(self):
        for adaptive in ['', 'adaptive_']:
            for num_dim in [1, 2, 3]:
                fn_name = '{}max_pool{}d'.format(adaptive, num_dim)
                fn = getattr(F, fn_name)
                x = torch.full([1, 1] + num_dim * [3], nan, device=device)
                res = fn(x, 1 if adaptive else 3)
                self.assertTrue(math.isnan(res.item()))

    def test_pool_large_size(self):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # 16777217 is the smallest integer not expressible in float32
                x = torch.ones([1, 1, 16777217] + (num_dim - 1) * [1],
                               device=device)
                res = fn(x, 1, stride=1, padding=0)
                # check if the output shape was still computed correctly
                self.assertEqual(x.shape[2], res.shape[2])

    def test_pool_invalid_size(self):
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = '{}_pool{}d'.format(op, num_dim)
                fn = getattr(F, fn_name)
                # use a configuration that gives zero outputs only
                # when doing a correct floor division by the stride
                x = torch.ones([1, 1] + num_dim * [4],
                               device=device)

                try:
                    fn(x, 3, stride=2, padding=0, dilation=2)
                except Exception as e:
                    try:
                        fn(x, 6, stride=2, padding=0)
                    except Exception as e:
                        self.assertIn("too small", str(e))

class TestConv(TestCase):
    def run_conv_double_back_test(self, kern, stride, padding, chan_in, chan_out, batch_size,
                                  inp_size, dilation, no_weight, groups=1, use_cuda=False,
                                  use_bias=True, dtype=torch.double):
        device = torch.device(device)
        x = torch.randn(batch_size, chan_in, inp_size, inp_size, device=device,
                        dtype=dtype, requires_grad=True)
        weight = torch.randn(chan_out, chan_in // groups, kern, kern, device=device,
                             dtype=dtype, requires_grad=not no_weight)
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
        else:
            bias = None

        def func(*inputs):
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            # We disable cudnn during forward to avoid finite difference imprecision issues
            with cudnn.flags(enabled=False):
                out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        dummy_out = func(*inputs)
        grad_y = torch.randn_like(dummy_out, device=device, dtype=dtype, requires_grad=True)

        # Issue #15353: test mkldnn double backward, don't run gradgradcheck due
        # to imprecision issues
        if dtype == torch.float:
            g, = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad

        return gradgradcheck(func, inputs, (grad_y,))

    def test_noncontiguous_weight(self):
        # Noncontiguous weights must be contiguous() before being
        #device = 'cpu:0'
        input = torch.tensor([1, 1, 1], dtype=torch.float32, device=device).view(1, 1, 3)
        weights1 = torch.tensor([1], dtype=torch.float32, device=device).expand(1, 1, 2)
        weights2 = torch.tensor([1], dtype=torch.float32, device=device).expand(1, 1, 2).contiguous()
        res1 = F.conv1d(input, weights1, bias=None, stride=2, dilation=2)
        res2 = F.conv1d(input, weights2, bias=None, stride=2, dilation=2)

        res1_clone = res1.clone().view(res1.numel())
        res2_clone = res2.clone().view(res2.numel())
        self.assertEqual(res1.size(), res2.size())
        for i in range(0, res2.numel()):
           self.assertEqual(res1_clone[i], res2_clone[i])

    def test_mismatch_shape_conv2d(self):
        x = torch.randn(1, 10, 1, 28, 28, device=device)
        w = torch.randn(6, 1, 5, 5, device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    r'Expected 4-dimensional input for 4-dimensional weight \[6, 1, 5, 5\],' +
                                    r' but got 5-dimensional input of size \[1, 10, 1, 28, 28\] instead'):

            F.conv2d(x, w)

    def test_Conv2d_with_cpu(self):
        conv_dpcpp = torch.nn.Conv2d(3, 3, 3).to(device=device)
        conv_cpu = torch.nn.Conv2d(3, 3, 3)
        inputs_cpu = torch.randn(2, 3, 5, 5, requires_grad=True)
        inputs_dpcpp = inputs_cpu.detach().to(device=device).requires_grad_(True)
        conv_dpcpp.bias.data = conv_cpu.bias.data.to(device=device)
        conv_dpcpp.weight.data = conv_cpu.weight.data.to(device=device)

        out_dpcpp = conv_dpcpp(inputs_dpcpp)
        out_cpu = conv_cpu(inputs_cpu)
        self.assertEqual(out_dpcpp.to('cpu'), out_cpu, prec=1e-4)
        out_dpcpp.sum().backward()
        out_cpu.sum().backward()
        self.assertEqual(inputs_dpcpp.grad.to('cpu'), inputs_cpu.grad, prec=1e-4)

if __name__ == '__main__':
    test = unittest.main()
