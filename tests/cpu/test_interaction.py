import unittest
import torch
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
            offset = 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
            return R

        dtypes = [torch.float32, torch.bfloat16]
        feature_sizes = [127, 128]
        for dtype, feature_size in itertools.product(dtypes, feature_sizes):
            x1 = (
                torch.randn([2048, feature_size])
                .to(dtype)
                .clone()
                .detach()
                .requires_grad_()
            )
            x2 = x1.clone().detach().requires_grad_()
            ly1 = []
            ly2 = []
            for i in range(0, 26):
                V = (
                    torch.randn([2048, feature_size])
                    .to(dtype)
                    .clone()
                    .detach()
                    .requires_grad_()
                )
                ly1.append(V)
                ly2.append(V.clone().detach().requires_grad_())

            A = interact_fusion(x1, ly1)
            B = interact_features(x2, ly2)
            if dtype == torch.bfloat16:
                """
                For inference:
                Mismatched elements: 17 / 978944 (0.0%)
                Greatest absolute difference: 0.0625 at index (1833, 223) (up to 0.0001 allowed)
                Greatest relative difference: 0.00775146484375 at index (1924, 160) (up to 0.0001 allowed)
                For training (grad compare):
                Mismatched elements: 73917 / 260096 (28.4%)
                Greatest absolute difference: 0.125 at index (52, 52)
                Greatest relative difference: inf at index (16, 58)
                There should be 2 reason impact the results here:
                (1): Our bf16 gemm will cast to FP32 first and use fp32 fma, onednn gemm
                might use amx: __tile_dpbf16ps or avx512_bf16: _mm512_dpbf16_ps which
                will compute first dot product at bf16 level
                for example:
                    tmp.fp32[n] += FP32(src0.row[m].bf16[2*k+0]) * FP32(src1.row[k].bf16[2*n+0])
                    tmp.fp32[n] += FP32(src0.row[m].bf16[2*k+1]) * FP32(src1.row[k].bf16[2*n+1])
                (2): For backward, we have an optimization to optimize
                A mm B + A mm C -> A mm (B + C), this change will bring some differences.
                """
                rtol, atol = 0.005, 0.1
            else:
                rtol, atol = None, None
            torch.testing.assert_allclose(A, B, rtol=rtol, atol=atol)
            A.sum().backward()
            B.sum().backward()
            torch.testing.assert_allclose(x1.grad, x2.grad, rtol=rtol, atol=atol)
            for i in range(0, 26):
                torch.testing.assert_allclose(
                    ly1[i].grad, ly2[i].grad, rtol=rtol, atol=atol
                )


if __name__ == "__main__":
    test = unittest.main()
