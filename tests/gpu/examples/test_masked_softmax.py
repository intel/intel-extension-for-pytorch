import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_masked_softmax_BL(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for B, num_heads, L in sizes:
            for dim in [0, 3]:
                input = torch.randn((B, num_heads, L, L))
                mask = torch.randint(0, 2, (B, L))
                mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L).bool()
                mask_type = 1  # BxL => src_key_padding_mask
                input = input.xpu()
                mask = mask.xpu()
                native_res = torch._masked_softmax(input, mask, dim, mask_type)
                mask = ~mask

                def slow_masked_softmax(input, mask):
                    exp = torch.exp(input)
                    exp = exp * mask
                    s = exp.sum(dim=dim, keepdim=True).expand(exp.size())
                    return exp / s

                pt_res = slow_masked_softmax(input, mask)
                pt_res = torch.nan_to_num(pt_res)

                mask_not = mask.logical_not()
                # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                # Converts rows with all True's to False
                mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                self.assertEqual(
                    pt_res.masked_fill(mask_out, 0),
                    native_res.masked_fill(mask_out, 0),
                    exact_dtype=True,
                )

    def test_masked_softmax_LL(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for B, num_heads, L in sizes:
            input = torch.randn((B, num_heads, L, L))
            mask = torch.randint(0, 2, (L, L)).bool()
            mask_type = 0  # LxL => src_mask
            dim = -1
            input = input.xpu()
            mask = mask.xpu()
            native_res = torch._masked_softmax(input, mask, dim, mask_type)
            pt_res = torch.softmax(input.masked_fill(mask, -float("inf")), dim)
            self.assertEqual(native_res.cpu(), pt_res.cpu())

    def _test_masked_softmax_helper(self, input, dim, mask, mask_type):
        input_ref = input.detach().clone().requires_grad_()
        result = torch._masked_softmax(input, mask, dim, mask_type)

        expected = torch._softmax(
            input_ref.masked_fill(mask, float("-inf")), dim, False
        )
        grad = torch.randn_like(expected).to(dtype=expected.dtype)

        result.backward(grad)
        expected.backward(grad)
        self.assertEqual(input.grad, torch.nan_to_num(input_ref.grad))
        self.assertEqual(input.grad, input.grad.masked_fill(mask, 0.0))

        # Make sure the optional argument works as well
        if dim == input.dim() - 1:
            input_ref_default = input.detach().clone().requires_grad_()
            result_default = torch._masked_softmax(
                input_ref_default, mask, None, mask_type
            )
            result_default.backward(grad)
            self.assertEqual(result, result_default)
            self.assertEqual(input.grad, input_ref_default.grad)

        # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
        # Converts rows with all True's to False
        mask_out = mask.all(dim, keepdim=True).expand(mask.shape)
        self.assertEqual(
            result.masked_fill(mask_out, 0), expected.masked_fill(mask_out, 0)
        )

    def test_masked_softmax_grad(self):
        shapes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for shape in shapes:
            dims = [0, len(shape) - 1] if len(shape) > 0 else [0]
            for dim in dims:
                for mask_type in [1, 2]:  # 1 = BxL => src_key_padding_mask
                    input = torch.randn(shape, requires_grad=True)
                    mask = torch.randint(0, 2, shape).bool()
                    input = input.xpu().detach().requires_grad_()
                    mask = mask.xpu()
                    self._test_masked_softmax_helper(input, dim, mask, mask_type)
