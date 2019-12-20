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

class RN50(TestCase):
    def test_ones(self):
        # test boolean tensor
        res1 = torch.ones(1, 2, dtype=torch.bool, device='dpcpp:0')
        expected = torch.tensor([[True, True]], dtype=torch.bool, device='dpcpp:0')
        self.assertEqual(res1, expected)

if __name__ == '__main__':
    test = unittest.main()
