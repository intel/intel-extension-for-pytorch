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

from common_utils import (
    TestCase, TEST_WITH_ROCM, run_tests,
    IS_WINDOWS, IS_FILESYSTEM_UTF8_ENCODING, NO_MULTIPROCESSING_SPAWN,
    do_test_dtypes, IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, load_tests, slowTest,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, make_tensor)

class TestInteractionCases(TestCase):
    def test_interaction(self):
        def interact_fusion(x, ly):
            A = [x] + ly
            R = ipex.interaction(*A)
            return R

        def interact_features(x, ly):
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # Z = pcl_embedding_bag.bdot(T)
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset =  0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)], device=ipex.DEVICE)
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)], device=ipex.DEVICE)
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
            return R

        dtypes=[torch.float32]
        for dtype in dtypes:
            x1 = torch.randn([2048, 128], device=ipex.DEVICE).to(dtype).clone().detach().requires_grad_()
            x2 = x1.clone().detach().requires_grad_()
            ly1 = []
            ly2 = []
            for i in range(0, 26):
                V = torch.randn([2048, 128], device=ipex.DEVICE).to(dtype).clone().detach().requires_grad_()
                ly1.append(V)
                ly2.append(V.clone().detach().requires_grad_())

            A = interact_fusion(x1, ly1)
            B = interact_features(x2, ly2)
            self.assertEqual(A, B)

            A.mean().backward()
            B.mean().backward()
            self.assertEqual(x1.grad, x2.grad)
            for i in range(0, 26):
                self.assertEqual(ly1[i].grad, ly2[i].grad)

if __name__ == '__main__':
    test = unittest.main()
