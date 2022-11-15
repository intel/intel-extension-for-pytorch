import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_real(self, dtype=torch.cfloat):
        x = torch.randn(3, 4, 5, dtype=dtype)
        x_xpu = x.to("xpu")
        self.assertEqual(x.real, x_xpu.real.cpu())
