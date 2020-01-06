"""Tests for xla_dist."""
from __future__ import division
from __future__ import print_function

import sys
import io
import os
import math
import random
import re
import copy
import shutil
import uuid
import unittest
from unittest import mock

import torch
import intel_pytorch_extension as ipex
import torch.nn.functional as F
from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, brute_pdist, brute_cdist, slowTest, \
    skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf


device = 'dpcpp:0'
SIZE = 100

class RN50(TestCase):
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

    def test_nll_loss_mismatched_batch(self):
        x = torch.randn((10, 3), device=device, requires_grad=True)
        # t should have size (10,)
        t = torch.zeros((3,), device=device, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

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

if __name__ == '__main__':
    test = unittest.main()
