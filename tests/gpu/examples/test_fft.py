import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest


@pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
class TestNNMethod(TestCase):
    def test_fft(self, dtype=torch.float):
        x1 = torch.randn(5, 5)
        x1_dpcpp = x1.to("xpu")

        y1 = torch.stft(x1, 1, hop_length=1)
        y1_dpcpp = torch.stft(x1_dpcpp, 1, hop_length=1)
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
        x = torch.randn(5, 6, 7)
        x_dpcpp = x.to("xpu")

        for dim in range(-len(x.size()), 0):
            y1 = torch.fft.fftn(x, dim=dim)
            y2 = torch.fft.ifftn(y1, dim=dim)

            y1_dpcpp = torch.fft.fftn(x, dim=dim)
            y2_dpcpp = torch.fft.ifftn(y1_dpcpp, dim=dim)

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
