import torch
import torch_ipex
from torch.testing._internal.common_utils import TestCase
import pytest

class TestNNMethod(TestCase):
    @pytest.mark.skipif("not torch_ipex._onemkl_is_enabled()")
    def test_fft(self, dtype=torch.float):
        x1 = torch.randn(5, 5)
        x2 = torch.randn(4, 3, 2)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        y1 = torch.rfft(x1, 2, onesided=True, normalized=False)
        y3 = torch.stft(x1, 1, hop_length=1)
        y4 = torch.fft(x2, 2)
        y5 = torch.ifft(y4, 2)

        y1_dpcpp = torch.rfft(x1_dpcpp, 2, onesided=True, normalized=False)
        y3_dpcpp = torch.stft(x1_dpcpp, 1, hop_length=1)
        y4_dpcpp = torch.fft(x2_dpcpp, 2)
        y5_dpcpp = torch.ifft(y4_dpcpp, 2)


        self.assertEqual(y1, y1_dpcpp.cpu())
        self.assertEqual(y3, y3_dpcpp.cpu())
        self.assertEqual(y4, y4_dpcpp.cpu())
        self.assertEqual(y5, y5_dpcpp.cpu())

    @pytest.mark.skipif(reason="irfft has accuracy issue in oneMKL")
    def test_irfft(self, dtype=torch.float):
        x1 = torch.randn(5, 5)
        x2 = torch.randn(4, 3, 2)
        x1_dpcpp = x1.to("xpu")
        x2_dpcpp = x2.to("xpu")
        y1 = torch.rfft(x1, 2, onesided=True, normalized=False)
        y2 = torch.irfft(y1, 2, onesided=True, signal_sizes=x1.shape)

        y1_dpcpp = torch.rfft(x1_dpcpp, 2, onesided=True, normalized=False)
        y2_dpcpp = torch.irfft(y1_dpcpp, 2, onesided=True, signal_sizes=x1.shape)

        self.assertEqual(y2, y2_dpcpp.cpu())
