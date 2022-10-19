import torch
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)
import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @repeat_test_for_types([torch.float, torch.double])
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_matrix_exp(self, dtype=torch.float):
        A = torch.randn([10, 20, 20], dtype=dtype)
        out_cpu = torch.matrix_exp(A)
        A_xpu = A.to("xpu")
        out_xpu = torch.matrix_exp(A_xpu)
        self.assertEqual(out_cpu, out_xpu.cpu())

    @repeat_test_for_types([torch.float, torch.double])
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_matrix_exp_complex(self, dtype=torch.float):
        real = torch.randn([10, 20, 20], dtype=dtype)
        imag = torch.randn([10, 20, 20], dtype=dtype)
        A = torch.complex(real, imag)
        out_cpu = torch.matrix_exp(A)
        A_xpu = A.to("xpu")
        out_xpu = torch.matrix_exp(A_xpu)
        self.assertEqual(out_cpu, out_xpu.cpu(), rtol=1e-03, atol=1e-03)
