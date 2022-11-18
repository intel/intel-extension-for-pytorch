import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa F401
import pytest # noqa F401


class TestTorchMethod(TestCase):
    def test_lshift(self, dtype=torch.int32):
        x_cpu = torch.tensor([19, -20, -21, 22], dtype=dtype)
        y_cpu = torch.tensor([2, 1, 3, 1], dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_xpu = y_cpu.to("xpu")

        re_cpu = x_cpu.__lshift__(y_cpu)
        re_xpu = x_xpu.__lshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)

        re_cpu = x_cpu.__ilshift__(y_cpu)
        re_xpu = x_xpu.__ilshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)

    def test_rshift(self, dtype=torch.int32):
        x_cpu = torch.tensor([19, -20, -21, 22], dtype=dtype)
        y_cpu = torch.tensor([2, 1, 3, 1], dtype=dtype)
        x_xpu = x_cpu.to("xpu")
        y_xpu = y_cpu.to("xpu")

        re_cpu = x_cpu.__rshift__(y_cpu)
        re_xpu = x_xpu.__rshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)

        re_cpu = x_cpu.__irshift__(y_cpu)
        re_xpu = x_xpu.__irshift__(y_xpu).cpu()
        self.assertEqual(re_cpu, re_xpu)
