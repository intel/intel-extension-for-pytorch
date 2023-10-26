import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa F401
import pytest


@pytest.mark.skipif(not torch.xpu.has_onemkl(), reason="ipex build w/o oneMKL support")
@pytest.mark.skipif(not torch.has_mkl, reason="torch build w/o mkl support")
class TestTorchMethod(TestCase):
    def test_fft_float(self, dtype=torch.float):
        f_real = torch.randn(2, 72, 72)
        f_imag = torch.randn(2, 72, 72)
        axis = -1
        var = f_real + +1j * f_imag
        var = torch.fft.fft(var, dim=axis, norm="ortho")

        f_real_dpcpp = f_real.to("xpu")
        f_imag_dpcpp = f_imag.to("xpu")
        var_dpcpp = f_real_dpcpp + +1j * f_imag_dpcpp
        var_dpcpp = torch.fft.fft(var_dpcpp, dim=axis, norm="ortho")
        self.assertEqual(var, var_dpcpp)

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    def test_fft_double(self, dtype=torch.double):
        f_real = torch.randn(2, 72, 72).to(torch.double)
        f_imag = torch.randn(2, 72, 72).to(torch.double)
        axis = -1
        var = f_real + +1j * f_imag
        var = torch.fft.fft(var, dim=axis, norm="ortho")

        f_real_dpcpp = f_real.to("xpu")
        f_imag_dpcpp = f_imag.to("xpu")
        var_dpcpp = f_real_dpcpp + +1j * f_imag_dpcpp
        var_dpcpp = torch.fft.fft(var_dpcpp, dim=axis, norm="ortho")
        self.assertEqual(var, var_dpcpp)

    def test_fftn(self):
        x4D_cpu = torch.rand(1, 1, 1, 1, 1, device="cpu", dtype=torch.cfloat)
        out_cpu = torch.fft.fftn(x4D_cpu)
        x4D_xpu = x4D_cpu.to(device="xpu")
        out_xpu = torch.fft.fftn(x4D_xpu)
        self.assertEqual(out_cpu, out_xpu.cpu())
