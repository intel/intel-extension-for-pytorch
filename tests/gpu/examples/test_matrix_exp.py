import torch
from torch.testing._internal.common_utils import (TestCase)

from torch.testing._internal.common_device_type import (dtypes)

import intel_extension_for_pytorch # noqa


class TestTorchMethod(TestCase):
    @dtypes([torch.bfloat16, torch.float, torch.double])
    def test_matrix_exp(self, dtype=torch.bfloat16):
        A = torch.randn([10, 20, 20], dtype=dtype)
        out_cpu = torch.matrix_exp(A)
        A_xpu = A.to("xpu")
        out_xpu = torch.matrix_exp(A_xpu)
        self.assertEqual(out_cpu, out_xpu.cpu())

    @dtypes([torch.float, torch.double])
    def test_matrix_exp_complex(self, dtype=torch.float):
        real = torch.randn([10, 20, 20], dtype=dtype)
        imag = torch.randn([10, 20, 20], dtype=dtype)
        A = torch.complex(real, imag)
        out_cpu = torch.matrix_exp(A)
        A_xpu = A.to("xpu")
        out_xpu = torch.matrix_exp(A_xpu)
        self.assertEqual(out_cpu, out_xpu.cpu(), rtol=1e-03, atol=1e-03)
