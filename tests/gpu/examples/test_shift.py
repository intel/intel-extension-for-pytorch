import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_lshift(self, dtype=torch.float):
        x_cpu = torch.randn(4, dtype=torch.double)
        y_cpu = torch.randn(1, dtype=torch.double)
        x_xpu = x_cpu.to("xpu")
        y_xpu = y_cpu.to("xpu")

        re_cpu = x_cpu.__lshift__(y_cpu)
        re_xpu = x_xpu.__lshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)

        re_cpu = x_cpu.__ilshift__(y_cpu)
        re_xpu = x_xpu.__ilshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_rshift(self, dtype=torch.float):
        x_cpu = torch.randn(4, dtype=torch.double)
        y_cpu = torch.randn(1, dtype=torch.double)
        x_xpu = x_cpu.to("xpu")
        y_xpu = y_cpu.to("xpu")

        re_cpu = x_cpu.__rshift__(y_cpu)
        re_xpu = x_xpu.__rshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)

        re_cpu = x_cpu.__irshift__(y_cpu)
        re_xpu = x_xpu.__irshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)
