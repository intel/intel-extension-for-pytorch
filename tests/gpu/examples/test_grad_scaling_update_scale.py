import torch
import intel_extension_for_pytorch  # noqa
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_grad_scaling_update_scale(self, device="xpu", dtype=torch.float):
        growth = 2.0
        backoff = 0.25
        growth_interval = 2
        scale = torch.full((1,), 4.0, dtype=dtype, device=device)
        growth_tracker = torch.full((1,), 0.0, dtype=torch.int32, device=device)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="xpu")

        # Simulates 2 consecutive unskipped iterations
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 1)
        self.assertEqual(scale, 4.0)
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 8.0)

        # Simulates a skipped iteration
        found_inf.fill_(1.0)
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth, backoff, growth_interval)
        self.assertEqual(growth_tracker, 0)
        self.assertEqual(scale, 2.0)

    @repeat_test_for_types([torch.float, torch.half, torch.bfloat16])
    def test_grad_scaling_unscale(self, dtype=torch.float):
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device="xpu")
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device="xpu")

        size = 10
        g = torch.full((size, size), 4.0, dtype=dtype, device="xpu")
        ginf = g.clone()
        ginf[2, 2] = float('inf')
        gnan = g.clone()
        gnan[2, 2] = float('nan')

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # When passing lists with mismatched dtypes to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call,
        # it's expected to fall back to single-tensor TensorIterator kernel.
        grads = [g.clone(), g.to(dtype=torch.float16)]
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        for grad in grads:
            self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)
