import torch
from torch._six import inf
from torch.distributions import Geometric
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_dtype import all_types_and

import intel_extension_for_pytorch # noqa
import numpy as np
import pytest

from functools import wraps
# TODO : Rebasing 1.13. 1.13 retires the repeat_test_for_types.
# However, by the time submitting this change, the tests in experiments folder 
# are not handled correctly, thus, we put the function here for simplicity.
def repeat_test_for_types(dtypes):
    def repeat_helper(f):
        @wraps(f)
        def call_helper(self, *args):
            for dtype in dtypes:
                with TestCase.subTest(self, dtype=dtype):
                    f(self, *args, dtype=dtype)

        return call_helper
    return repeat_helper


cpu_device = torch.device("cpu")
sycl_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_geometric(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True, device=sycl_device)
        r = torch.tensor(0.3, requires_grad=True, device=sycl_device)
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        self.assertFalse(Geometric(p).sample().requires_grad)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r).sample().size(), ())
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        self.assertRaises(NotImplementedError, Geometric(r).rsample)


    def test_geometric_113(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True, device=sycl_device)
        r = torch.tensor(0.3, requires_grad=True, device=sycl_device)
        s = 0.3
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        self.assertEqual(Geometric(1).sample(), 0)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(1.)), -inf)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(0.)), 0)
        self.assertFalse(Geometric(p).sample().requires_grad)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r).sample().size(), ())
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        self.assertEqual(Geometric(s).sample().size(), ())
        # self._gradcheck_log_prob(Geometric, (p,))
        self.assertRaises(ValueError, lambda: Geometric(0))
        self.assertRaises(NotImplementedError, Geometric(r).rsample)

    @repeat_test_for_types([*all_types_and(torch.half, torch.bfloat16)])
    def test_geometric(self, dtype=torch.float):
        device = sycl_device
        # This function is directly ported from 1.13 test_torch.py
        a = torch.tensor([10], dtype=dtype, device=device).geometric_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))
    
    @pytest.mark.skip()
    @repeat_test_for_types([*all_types_and(torch.half, torch.bfloat16)])
    def test_geometric_kstest(self,  dtype):
        device = sycl_device
        print("Dtype is ", dtype, flush=True)

        from scipy import stats
        size = 1000
        for p in [0.2, 0.5, 0.8]:
            t = torch.empty(size, dtype=dtype, device=device).geometric_(p=p)
            actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
            expected = stats.geom(p).pmf(np.arange(1, 99)) * size

            res = stats.chisquare(actual, expected)
            self.assertEqual(res.pvalue, 1.0, atol=0.1, rtol=0)
