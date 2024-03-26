import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import pytest


def generate_fftn_test_param(dtype=torch.float):
    from itertools import islice, combinations

    fft_functions = [(torch.fft.fftn, torch.fft.ifftn)]

    lft, rgh = [5, 11]
    input_dims = [list(range(lft, rgh + 1))[:i] for i in range(3, (rgh + 1) - lft + 1)]

    for input_dim in input_dims:
        signal_dim = list(range(-len(input_dim), 0))
        signal_dims = [None]

        for r in islice(reversed(range(1, len(signal_dim))), 3):
            signal_dims.extend(islice(combinations(signal_dim, r=r), 3))

        yield input_dim, signal_dims, fft_functions


@pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
class TestNNMethod(TestCase):
    def test_fft(self, dtype=torch.float):
        x1 = torch.randn(5, 5)
        x1_dpcpp = x1.to("xpu")

        y1 = torch.stft(x1, 1, hop_length=1, return_complex=True)
        y1_dpcpp = torch.stft(x1_dpcpp, 1, hop_length=1, return_complex=True)
        self.assertEqual(y1, y1_dpcpp.cpu())

        x2 = torch.randn(5, 6, 7)
        x2_dpcpp = x2.to("xpu")
        for dim in range(-len(x2.size()), 0):
            y2 = torch.fft.fft(x2, dim=dim)
            y3 = torch.fft.ifft(y2, dim=dim)

            y2_dpcpp = torch.fft.fft(x2_dpcpp, dim=dim)
            y3_dpcpp = torch.fft.ifft(y2_dpcpp, dim=dim)

            self.assertEqual(y2, y2_dpcpp.cpu())
        self.assertEqual(y3, y3_dpcpp.cpu())

    def test_fftn(self, dtype=torch.float):
        from itertools import product

        for input_dim, signal_dims, fft_functions in generate_fftn_test_param(dtype):
            x = torch.randn(*input_dim, dtype=dtype)
            x_dpcpp = x.to("xpu")

            for dim, (fwd, bwd) in product(signal_dims, fft_functions):
                y1 = fwd(x, dim=dim)
                y2 = bwd(y1, dim=dim)

                y1_dpcpp = fwd(x_dpcpp, dim=dim)
                y2_dpcpp = bwd(y1_dpcpp, dim=dim)

                self.assertEqual(y2, y2_dpcpp.cpu())

    def test_irfft(self, dtype=torch.float):
        x1 = torch.randn(5, 5)
        x2 = torch.randn(4, 3, 2)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        y1 = torch.fft.rfft(x1, 2)
        y2 = torch.fft.irfft(y1, 2)

        y1_dpcpp = torch.fft.rfft(x1_dpcpp, 2)
        y2_dpcpp = torch.fft.irfft(y1_dpcpp, 2)

        self.assertEqual(y2, y2_dpcpp.cpu())

    def test_ifft2(self, dtype=torch.float):
        f_real = torch.randn(2, 72, 72)
        f_imag = torch.randn(2, 72, 72)
        axes = (1, 2)
        var = f_real + +1j * f_imag

        dst_cpu = torch.fft.ifft2(var, dim=axes)

        f_real_dpcpp = f_real.to("xpu")
        f_imag_dpcpp = f_imag.to("xpu")
        var_dpcpp = f_real_dpcpp + +1j * f_imag_dpcpp

        dst_dpcpp = torch.fft.ifft2(var_dpcpp, dim=axes)

        self.assertEqual(dst_cpu, dst_dpcpp.cpu())
