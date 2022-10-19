import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_dtype import (
    all_types, all_types_and
)
from itertools import product
import numpy as np
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_isin(self, device="xpu", dtype=torch.float):
        def assert_isin_equal(a, b):
            # Compare to the numpy reference implementation.
            x = torch.isin(a, b)
            a = a.cpu().numpy() if torch.is_tensor(a) else np.array(a)
            b = b.cpu().numpy() if torch.is_tensor(b) else np.array(b)
            y = np.isin(a, b)
            self.assertEqual(x, y)

        # multi-dim tensor, multi-dim tensor
        a = torch.arange(24, device=device, dtype=dtype).reshape([2, 3, 4])
        b = torch.tensor([[10, 20, 30], [0, 1, 3], [11, 22, 33]], device=device, dtype=dtype)
        assert_isin_equal(a, b)

        # zero-dim tensor
        zero_d = torch.tensor(3, device=device, dtype=dtype)
        assert_isin_equal(zero_d, b)
        assert_isin_equal(a, zero_d)
        assert_isin_equal(zero_d, zero_d)

        # empty tensor
        empty = torch.tensor([], device=device, dtype=dtype)
        assert_isin_equal(empty, b)
        assert_isin_equal(a, empty)
        assert_isin_equal(empty, empty)

        # scalar
        assert_isin_equal(a, 6)
        assert_isin_equal(5, b)

        def define_expected(lst, invert=False):
            expected = torch.tensor(lst, device=device)
            if invert:
                expected = expected.logical_not()
            return expected

        # Adapted from numpy's in1d tests
        for mult in [1, 10]:
            for invert in [False, True]:
                a = torch.tensor([5, 7, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)

                a[0] = 8
                ec = define_expected([False, False, True, True], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)

                a[0], a[3] = 4, 8
                ec = define_expected([True, False, True, False], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)

                a = torch.tensor([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], device=device, dtype=dtype)
                b = torch.tensor([2, 3, 4] * mult, device=device, dtype=dtype)
                ec = define_expected([False, True, False, True, True, True, True, True, True,
                                      False, True, False, False, False], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                b = torch.tensor([2, 3, 4] * mult + [5, 5, 4] * mult, device=device, dtype=dtype)
                ec = define_expected([True, True, True, True, True, True, True, True, True, True,
                                      True, False, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                a = torch.tensor([5, 7, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)

                # multi-dimensional input case using sort-based algo
                for assume_unique in [False, True]:
                    a = torch.arange(6, device=device, dtype=dtype).reshape([2, 3])
                    b = torch.arange(3, 30, device=device, dtype=dtype)
                    ec = define_expected([[False, False, False], [True, True, True]], invert=invert)
                    c = torch.isin(a, b, invert=invert, assume_unique=assume_unique)
                    self.assertEqual(c, ec)

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_isin_different_dtypes(self, device="xpu"):
        supported_types = all_types() if device == 'cpu' else all_types_and(torch.half)
        for mult in [1, 10]:
            for assume_unique in [False, True]:
                for dtype1, dtype2 in product(supported_types, supported_types):
                    a = torch.tensor([1, 2, 3], device=device, dtype=dtype1)
                    b = torch.tensor([3, 4, 5] * mult, device=device, dtype=dtype2)
                    ec = torch.tensor([False, False, True], device=device)
                    c = torch.isin(a, b, assume_unique=assume_unique)
                    self.assertEqual(c, ec)

    def test_isin_different_devices(self, device="xpu", dtype=torch.float):
        a = torch.arange(6, device=device, dtype=dtype).reshape([2, 3])
        b = torch.arange(3, 30, device='cpu', dtype=dtype)
        with self.assertRaises(RuntimeError):
            torch.isin(a, b)

        c = torch.arange(6, device='cpu', dtype=dtype).reshape([2, 3])
        d = torch.arange(3, 30, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            torch.isin(c, d)
