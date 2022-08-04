import torch
import intel_extension_for_pytorch # noqa
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
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
