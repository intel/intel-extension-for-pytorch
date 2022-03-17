import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest


@pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
class TestNNMethod(TestCase):
    @pytest.mark.skip(reason="not block the pre-ci")
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

    @pytest.mark.skip(reason="not block the pre-ci")
    def test_fft_bf16(self, dtype=torch.float):
        # Just for bf16 runtime test, there isn't cpu reference.
        var = torch.randn(2, 72, 72, 2).bfloat16()
        var_xpu = var.to("xpu")
        for i in range(2):
            dst_rfft = torch.rfft(var_xpu, 2, onesided=True, normalized=False)
            dst_ifft = torch.ifft(dst_rfft, signal_ndim=2, normalized=False)

    # TODO: remove skip when oneMKL is ready
    @pytest.mark.skip(reason="irfft has accuracy issue in oneMKL, MKLD-12824")
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
