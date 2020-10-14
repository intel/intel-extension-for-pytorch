import torch
import torch_ipex
from torch.testing._internal.common_utils import TestCase
import pytest

class TestNNMethod(TestCase):
    @pytest.mark.skip(reason='MKL support')
    def test_fft(self, dtype=torch.float):
        x = torch.randn(5, 5)
        x_dpcpp = x.to("dpcpp")
        y = torch.rfft(x, 2, onesided=True, normalized=False)
        y_dpcpp = torch.rfft(x_dpcpp, 2, onesided=True, normalized=False)

        self.assertEqual(y, y_dpcpp.cpu())

        y2 = torch.stft(x, 1, hop_length=1)
        y2_dpcpp = torch.stft(x, 1, hop_length=1)
        self.assertEqual(y2, y2_dpcpp.cpu())

