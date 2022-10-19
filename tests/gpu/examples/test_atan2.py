import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa
import pytest


class TestTorchMethod(TestCase):
    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_atan2(self, dtype=torch.float):

        x_cpu = torch.randn(4)
        y_cpu = torch.randn(4)

        dist_cpu = torch.atan2(x_cpu, y_cpu)

        x_dpcpp = x_cpu.to("xpu")
        y_dpcpp = y_cpu.to("xpu")

        dist_dpcpp = torch.atan2(x_dpcpp, y_dpcpp)

        self.assertEqual(x_cpu, x_dpcpp.cpu())
        self.assertEqual(y_cpu, y_dpcpp.cpu())
        self.assertEqual(dist_cpu, dist_dpcpp.cpu())

        x_cpu_int = torch.randint(100, (1, 10), dtype=torch.long)
        y_cpu_int = torch.randint(100, (1, 10), dtype=torch.long)

        dist_cpu_int = torch.atan2(x_cpu_int, y_cpu_int)

        x_dpcpp_int = x_cpu_int.to("xpu")
        y_dpcpp_int = y_cpu_int.to("xpu")

        dist_dpcpp_int = torch.atan2(x_dpcpp_int, y_dpcpp_int)

        self.assertEqual(x_cpu_int, x_dpcpp_int.cpu())
        self.assertEqual(y_cpu_int, y_dpcpp_int.cpu())
        self.assertEqual(dist_cpu_int, dist_dpcpp_int.cpu())
