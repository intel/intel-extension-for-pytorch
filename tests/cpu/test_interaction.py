import math
import random
import unittest
from functools import reduce

import torch

import intel_extension_for_pytorch as ipex

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

class TestInteractionCases(TestCase):
    def test_interaction(self):
        def interact_fusion(x, ly):
            A = [x] + ly
            R = ipex.nn.functional.interaction(*A)
            return R

        def interact_features(x, ly):
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset =  0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
            return R

        dtypes=[torch.float32, torch.bfloat16]
        for dtype in dtypes:
            x1 = torch.randn([2048, 128]).to(dtype).clone().detach().requires_grad_()
            x2 = x1.clone().detach().requires_grad_()
            ly1 = []
            ly2 = []
            for i in range(0, 26):
                V = torch.randn([2048, 128]).to(dtype).clone().detach().requires_grad_()
                ly1.append(V)
                ly2.append(V.clone().detach().requires_grad_())

            A = interact_fusion(x1, ly1)
            B = interact_features(x2, ly2)
            self.assertEqual(A, B)

            A.sum().backward()
            B.sum().backward()
            torch.testing.assert_allclose(x1.grad, x2.grad, rtol=0.005, atol=0.1)
            for i in range(0, 26):
                torch.testing.assert_allclose(ly1[i].grad, ly2[i].grad, rtol=0.005, atol=0.1)

if __name__ == '__main__':
    test = unittest.main()
