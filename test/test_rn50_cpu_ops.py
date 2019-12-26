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
from common_utils import TestCase, iter_indices, TEST_NUMPY, TEST_SCIPY, TEST_MKL, \
    TEST_LIBROSA, run_tests, download_file, skipIfNoLapack, suppress_warnings, \
    IS_WINDOWS, PY3, NO_MULTIPROCESSING_SPAWN, do_test_dtypes, do_test_empty_full, \
    IS_SANDCASTLE, load_tests, brute_pdist, brute_cdist, slowTest, \
    skipCUDANonDefaultStreamIf, skipCUDAMemoryLeakCheckIf


device = 'dpcpp:0'


def test_select():
    res1 = torch.ones(2, device=device)
    print(res1)


class RN50(TestCase):

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


if __name__ == '__main__':
    test = unittest.main()
