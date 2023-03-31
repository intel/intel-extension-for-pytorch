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


    def test_bessel_y0(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.bessel_y0(input0)
        result_xpu = torch.special.bessel_y0(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_bessel_y1(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.bessel_y1(input0)
        result_xpu = torch.special.bessel_y1(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_special_spherical_bessel_j0(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch._C._special.special_spherical_bessel_j0(input0)
        result_xpu = torch._C._special.special_spherical_bessel_j0(input0_xpu)

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

    def test_scaled_modified_bessel_k0(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.scaled_modified_bessel_k0(input0)
        result_xpu = torch.special.scaled_modified_bessel_k0(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_scaled_modified_bessel_k1(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.scaled_modified_bessel_k1(input0)
        result_xpu = torch.special.scaled_modified_bessel_k1(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_scaled_modified_bessel_k1(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.scaled_modified_bessel_k1(input0)
        result_xpu = torch.special.scaled_modified_bessel_k1(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)
    
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_log_ndtr(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        result_cpu = torch.special.log_ndtr(input0)
        result_xpu = torch.special.log_ndtr(input0_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_hermite_polynomial_he(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        input1_xpu = input1.clone().to("xpu")
        result_cpu = torch.special.hermite_polynomial_he(input0, input1)
        result_xpu = torch.special.hermite_polynomial_he(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_hermite_polynomial_h(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        input1_xpu = input1.clone().to("xpu")
        result_cpu = torch.special.hermite_polynomial_h(input0, input1)
        result_xpu = torch.special.hermite_polynomial_h(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_laguerre_polynomial_l(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        input1_xpu = input1.clone().to("xpu")
        result_cpu = torch.special.laguerre_polynomial_l(input0, input1)
        result_xpu = torch.special.laguerre_polynomial_l(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_legendre_polynomial_p(self, dtype=torch.float):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.clone().to("xpu")
        input1_xpu = input1.clone().to("xpu")
        result_cpu = torch.special.legendre_polynomial_p(input0, input1)
        result_xpu = torch.special.legendre_polynomial_p(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_chebyshev_polynomial_t(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        input1_xpu = input1.to("xpu")
        result_cpu = torch.special.chebyshev_polynomial_t(input0, input1)
        result_xpu = torch.special.chebyshev_polynomial_t(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_chebyshev_polynomial_u(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        input1_xpu = input1.to("xpu")
        result_cpu = torch.special.chebyshev_polynomial_u(input0, input1)
        result_xpu = torch.special.chebyshev_polynomial_u(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_chebyshev_polynomial_v(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        input1_xpu = input1.to("xpu")
        result_cpu = torch.special.chebyshev_polynomial_v(input0, input1)
        result_xpu = torch.special.chebyshev_polynomial_v(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_chebyshev_polynomial_w(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input1 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        input1_xpu = input1.to("xpu")
        result_cpu = torch.special.chebyshev_polynomial_w(input0, input1)
        result_xpu = torch.special.chebyshev_polynomial_w(input0_xpu, input1_xpu)

        self.assertEqual(result_xpu.to("cpu"), result_cpu)

    def test_shifted_chebyshev_polynomial_t(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        result_cpu = torch.special.shifted_chebyshev_polynomial_t(input0, input0)
        result_xpu = torch.special.shifted_chebyshev_polynomial_t(input0_xpu, input0_xpu)
        result_xpu = result_xpu.to("cpu")

        self.assertTrue(result_xpu.equal(result_cpu))

    def test_shifted_chebyshev_polynomial_u(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        result_cpu = torch.special.shifted_chebyshev_polynomial_u(input0, input0)
        result_xpu = torch.special.shifted_chebyshev_polynomial_u(input0_xpu, input0_xpu)
        result_xpu = result_xpu.to("cpu")

        self.assertTrue(result_xpu.equal(result_cpu))

    def test_shifted_chebyshev_polynomial_v(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        result_cpu = torch.special.shifted_chebyshev_polynomial_v(input0, input0)
        result_xpu = torch.special.shifted_chebyshev_polynomial_v(input0_xpu, input0_xpu)
        result_xpu = result_xpu.to("cpu")

        self.assertTrue(result_xpu.equal(result_cpu))

    def test_shifted_chebyshev_polynomial_w(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        result_cpu = torch.special.shifted_chebyshev_polynomial_w(input0, input0)
        result_xpu = torch.special.shifted_chebyshev_polynomial_w(input0_xpu, input0_xpu)
        result_xpu = result_xpu.to("cpu")

        self.assertTrue(result_xpu.equal(result_cpu))

    def test_special_airy_ai_out(self):
        input0 = torch.randn(8192, 8192, device="cpu")
        input0_xpu = input0.to("xpu")
        result_cpu = torch.special.airy_ai(input0)
        result_xpu = torch.special.airy_ai(input0_xpu)
        result_xpu = result_xpu.to("cpu")

        self.assertEqual(result_cpu, result_xpu, atol=1e-4, rtol=1e-5)
