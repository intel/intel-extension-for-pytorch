import torch
import intel_extension_for_pytorch # noqa
from scipy import special
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):

    def test_erfcx(self, dtype=torch.float):
        dtypes = [torch.float32, torch.float64, torch.bfloat16]
        device = "xpu"
        for dtype in dtypes:

            def check_equal(t, torch_fn, scipy_fn):
                # Test by comparing to scipy
                actual = torch_fn(t)
                if dtype is torch.bfloat16:
                    t = t.to(torch.float32)
                expected = scipy_fn(t.cpu().numpy())

                # Casting down for dtype float16 is required since scipy upcasts to float32
                if dtype is torch.bfloat16 or dtype is torch.float16:
                    expected = torch.from_numpy(expected).to(dtype)
                self.assertEqual(actual, expected)

            t = torch.tensor([], device=device, dtype=dtype)
            check_equal(t, torch.special.erfcx, special.erfcx)

            range = (-1e7, 1e7)

            t = torch.linspace(*range, int(1e4), device=device, dtype=dtype)
            check_equal(t, torch.special.erfcx, special.erfcx)

            # NaN, inf, -inf are tested in reference_numerics tests.
            info = torch.finfo(dtype)
            min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
            t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
            check_equal(t, torch.special.erfcx, special.erfcx)
