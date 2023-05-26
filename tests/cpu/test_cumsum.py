import torch
import unittest
from common_utils import TestCase


class TestCumSum(TestCase):
    # Port from test_torch
    def test_cumsum(self):
        for dtype in [torch.float, torch.double, torch.long]:
            x = torch.randn(17, 4097).to(dtype)
            res1 = torch.ops.torch_ipex.cumsum(x, 1)
            res2 = torch.tensor([]).to(dtype)
            torch.ops.torch_ipex.cumsum(x, 1, out=res2)
            self.assertEqual(res1, res2)
            torch.ops.torch_ipex.cumsum_(x, 1)
            self.assertEqual(res1, x)

        a = torch.tensor(
            [[True, False, True], [False, False, False], [True, True, True]]
        )
        b = a.byte()
        aRes = torch.ops.torch_ipex.cumsum(a, 0)
        bRes = torch.ops.torch_ipex.cumsum(b, 0)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 1], [1, 0, 1], [2, 1, 2]]))

        aRes = torch.ops.torch_ipex.cumsum(a, 1)
        bRes = torch.ops.torch_ipex.cumsum(b, 1)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 1, 2], [0, 0, 0], [1, 2, 3]]))

        # Check that cummulative sum over a zero length dimension doesn't crash on backprop.
        # Also check that cumsum over other dimensions in a tensor with a zero-length
        # dimensiuon also works
        # Also include a basic suite of similar tests for other bases cases.
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                integrated = torch.ops.torch_ipex.cumsum(raw_tensor, dim=dim)
                # Check that backward does not crash
                integrated.sum().backward()
                # Check that output maintained correct shape
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # Check a scalar example
        raw_tensor = torch.tensor(3.0, requires_grad=True)
        integrated = raw_tensor.cumsum(dim=-1)
        self.assertEqual(raw_tensor, integrated)
        # Check that backward does not crash
        integrated.sum().backward()
        # Check that output maintained correct shape
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)


if __name__ == "__main__":
    test = unittest.main()
