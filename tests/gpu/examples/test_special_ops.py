import torch
import intel_extension_for_pytorch  # noqa
from scipy import special
from torch.testing._internal.common_utils import TestCase
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
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

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_ndtri_entr(self, dtype=torch.float):
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
            check_equal(t, torch.special.ndtri, special.ndtri)
            check_equal(t, torch.special.entr, special.entr)

            range = (-1e7, 1e7)

            t = torch.linspace(*range, int(1e4), device=device, dtype=dtype)
            check_equal(t, torch.special.ndtri, special.ndtri)
            check_equal(t, torch.special.entr, special.entr)

            # NaN, inf, -inf are tested in reference_numerics tests.
            info = torch.finfo(dtype)
            min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
            t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
            check_equal(t, torch.special.ndtri, special.ndtri)
            check_equal(t, torch.special.entr, special.entr)

    def test_bessel_j0(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.bessel_j0(input0)
        result_xpu = torch.special.bessel_j0(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_bessel_j1(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.bessel_j1(input0)
        result_xpu = torch.special.bessel_j1(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_modified_bessel_i0(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.modified_bessel_i0(input0)
        result_xpu = torch.special.modified_bessel_i0(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_modified_bessel_i1(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.modified_bessel_i1(input0)
        result_xpu = torch.special.modified_bessel_i1(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_modified_bessel_k0(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.modified_bessel_k0(input0)
        result_xpu = torch.special.modified_bessel_k0(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_modified_bessel_k1(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.modified_bessel_k1(input0)
        result_xpu = torch.special.modified_bessel_k1(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

