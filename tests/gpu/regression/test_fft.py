import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)
import intel_extension_for_pytorch
import pytest

class TestTorchMethod(TestCase):
    @pytest.mark.skipif(torch.xpu.device_count()==1, reason="doesn't support with one device")
    @repeat_test_for_types([torch.float, torch.double])
    def test_fft(self, dtype=torch.float):
        f_real = torch.randn(2, 72, 72)
        f_imag = torch.randn(2, 72, 72)
        axis = -1
        var = f_real + + 1j * f_imag
        var = torch.fft.fft(var, dim=axis, norm='ortho')

        f_real_dpcpp = f_real.to("xpu")
        f_imag_dpcpp = f_imag.to("xpu")
        var_dpcpp = f_real_dpcpp + + 1j * f_imag_dpcpp
        var_dpcpp = torch.fft.fft(var_dpcpp, dim=axis, norm='ortho')
        self.assertEqual(var, var_dpcpp)

