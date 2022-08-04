import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

import pytest


@pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
class TestNNMethod(TestCase):
    def test_fft(self, dtype=torch.float):
        x1 = torch.randn(5, 5)
        x2 = torch.randn(4, 3, 2)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        y1 = torch.fft.rfft(x1, 2)
        y3 = torch.stft(x1, 1, hop_length=1)
        y4 = torch.fft.fft(x2, 2)
        y5 = torch.fft.ifft(y4, 2)

        y1_dpcpp = torch.fft.rfft(x1_dpcpp, 2)
        y3_dpcpp = torch.stft(x1_dpcpp, 1, hop_length=1)
        y4_dpcpp = torch.fft.fft(x2_dpcpp, 2)
        y5_dpcpp = torch.fft.ifft(y4_dpcpp, 2)

        self.assertEqual(y1, y1_dpcpp.cpu())
        self.assertEqual(y3, y3_dpcpp.cpu())
        self.assertEqual(y4, y4_dpcpp.cpu())
        self.assertEqual(y5, y5_dpcpp.cpu())

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
