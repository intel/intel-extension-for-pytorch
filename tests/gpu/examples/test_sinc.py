import torch
from torch.autograd import gradcheck
from torch.testing._internal.common_utils import (TestCase,
                                                  repeat_test_for_types)

import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_sinc_with_grad(self):
        # The derivative of sinc(x) at x=0 has to be special cased.
        # A naive computation will result in 0/0 -> NaN.
        # We also need to be careful when we are very close to 0, as the
        # derivative's denominator is squared, and there are some floats
        # that are positive and whose squares are zero.
        a = torch.tensor([0.0, torch.finfo(torch.double).tiny, 1.0],
                         dtype=torch.double,
                         requires_grad=True)

        a_xpu = a.clone().to(dpcpp_device)
        gradcheck(torch.sinc, a_xpu)

    @repeat_test_for_types([torch.float, torch.bfloat16])
    def test_sinc(self, dtype):
        a = torch.randn(4, device=cpu_device, dtype=dtype)
        a_xpu = a.clone().to(dpcpp_device)
        result_cpu = torch.sinc(a)
        result_xpu = torch.sinc(a_xpu)
        special_xpu = torch.special.sinc(a_xpu)
        self.assertEqual(result_cpu, result_xpu.cpu())
        self.assertEqual(special_xpu.cpu(), result_xpu.cpu())
