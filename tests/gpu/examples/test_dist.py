import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_dist(self, dtype=torch.float32):
        x_cpu = torch.randn(4)
        y_cpu = torch.randn(4)
        x_dpcpp = x_cpu.to("xpu")
        y_dpcpp = y_cpu.to("xpu")

        z1_cpu = torch.dist(x_cpu, y_cpu, 3.5)
        z1_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 3.5)
        self.assertEqual(z1_cpu, z1_dpcpp.cpu())

        z2_cpu = torch.dist(x_cpu, y_cpu, 3)
        z2_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 3)
        self.assertEqual(z2_cpu, z2_dpcpp.cpu())

        z3_cpu = torch.dist(x_cpu, y_cpu, 0)
        z3_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 0)
        self.assertEqual(z3_cpu, z3_dpcpp.cpu())

        z4_cpu = torch.dist(x_cpu, y_cpu, 1)
        z4_dpcpp = torch.dist(x_dpcpp, y_dpcpp, 1)
        self.assertEqual(z4_cpu, z4_dpcpp.cpu())

        self.assertEqual(x_cpu.dist(y_cpu, 3.5), x_dpcpp.dist(y_dpcpp, 3.5).cpu())
        self.assertEqual(x_cpu.dist(y_cpu, 3), x_dpcpp.dist(y_dpcpp, 3).cpu())
        self.assertEqual(x_cpu.dist(y_cpu, 0), x_dpcpp.dist(y_dpcpp, 0).cpu())
        self.assertEqual(x_cpu.dist(y_cpu, 1), x_dpcpp.dist(y_dpcpp, 1).cpu())
