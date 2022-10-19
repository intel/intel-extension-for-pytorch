import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    def test_bincount(self, dtype=torch.float):
        x_cpu = torch.randint(0, 8, (5,), dtype=torch.int64)
        y_cpu = torch.linspace(0, 1, steps=5)
        x_dpcpp = x_cpu.to("xpu")
        y_dpcpp = y_cpu.to("xpu")
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(torch.bincount(x_cpu),
                         torch.bincount(x_dpcpp).cpu())

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_histc(self, dtype=torch.float):
        x_cpu = torch.randint(0, 8, (5,), dtype=torch.double)
        x_dpcpp = x_cpu.to("xpu")
        res = torch.histc(x_cpu, bins=4, min=0, max=3)
        res_dpcpp = torch.histc(x_dpcpp, bins=4, min=0, max=3)
        res_tensor = x_cpu.histc(bins=4, min=0, max=3)
        res_tensor_dpcpp = x_dpcpp.histc(bins=4, min=0, max=3)
        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(res, res_dpcpp.cpu())
        self.assertEqual(res_tensor, res_tensor_dpcpp.cpu())
