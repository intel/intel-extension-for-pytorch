import unittest
import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from common_utils import TestCase
import itertools

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

        dtypes = [torch.float32, torch.bfloat16]
        feature_sizes = [127, 128]
        for dtype, feature_size in itertools.product(dtypes, feature_sizes):
            x1 = torch.randn([2048, feature_size]).to(dtype).clone().detach().requires_grad_()
            x2 = x1.clone().detach().requires_grad_()
            ly1 = []
            ly2 = []
            for i in range(0, 26):
                V = torch.randn([2048, feature_size]).to(dtype).clone().detach().requires_grad_()
                ly1.append(V)
                ly2.append(V.clone().detach().requires_grad_())

            A = interact_fusion(x1, ly1)
            B = interact_features(x2, ly2)
            # For FP32 data type, fused interaction will use MKLDNN gemm while
            # non-fused interaction will use GEMM. So there might be a small difference here
            torch.testing.assert_allclose(A, B, rtol=1e-4, atol=1e-4)

            A.sum().backward()
            B.sum().backward()
            torch.testing.assert_allclose(x1.grad, x2.grad, rtol=0.005, atol=0.1)
            for i in range(0, 26):
                torch.testing.assert_allclose(ly1[i].grad, ly2[i].grad, rtol=0.005, atol=0.1)

if __name__ == '__main__':
    test = unittest.main()
