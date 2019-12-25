"""Tests for xla_dist."""
from __future__ import division
from __future__ import print_function

import uuid
import unittest
from unittest import mock

import torch
import intel_pytorch_extension as ipex
from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, brute_pdist, brute_cdist, slowTest, \
    skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf

#str_device_const = 'cpu:0'
str_device_const = 'dpcpp:0'

class RN50(TestCase):
    def test_ones(self):
        res1 = torch.ones(3, device=str_device_const)
        mask = torch.tensor([True, False, True], device=str_device_const)
        res1[mask] = 3
        res2 = torch.tensor([3.0, 1.0, 3.0], device=str_device_const)
        self.assertEqual(res1, res2)

    def test_add(self):
        m1 = torch.randn(2, 3, device=str_device_const)
        v1 = torch.randn(3, device=str_device_const)

        # contiguous
        res1 = torch.add(m1[1], v1)
        res2 = torch.zeros(3, device=str_device_const)
        for i in range(m1.size(1)):
            res2[i] = m1[1, i] + v1[i]
            print('{} {}'.format(str(res1[i]), str(res2[i])))
        self.assertEqual(res1, res2)

if __name__ == '__main__':
    test = unittest.main()
