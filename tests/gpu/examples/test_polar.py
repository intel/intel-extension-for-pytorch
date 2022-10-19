import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_polar_float(self, dtype=torch.float):
        abs_cpu = torch.randn([5, 5])
        angle_cpu = torch.randn([5, 5])
        y_cpu = torch.polar(abs_cpu, angle_cpu)
        abs_xpu = abs_cpu.to("xpu")
        angle_xpu = angle_cpu.to("xpu")
        y_xpu = torch.polar(abs_xpu, angle_xpu)

        self.assertEqual(y_cpu, y_xpu.to("cpu"))

    @pytest.mark.skipif(not torch.xpu.utils.has_fp64_dtype(), reason="fp64 not support by this device")
    def test_polar_double(self, dtype=torch.double):
        abs_cpu = torch.randn([5, 5], dtype=dtype)
        angle_cpu = torch.randn([5, 5], dtype=dtype)
        y_cpu = torch.polar(abs_cpu, angle_cpu)
        abs_xpu = abs_cpu.to("xpu")
        angle_xpu = angle_cpu.to("xpu")
        y_xpu = torch.polar(abs_xpu, angle_xpu)

        self.assertEqual(y_cpu, y_xpu.to("cpu"))
