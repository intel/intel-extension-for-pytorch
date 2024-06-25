import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_dot(self, dtype=torch.float):
        x1_cpu = torch.rand(10000, dtype=torch.float)
        x2_cpu = torch.rand(10000, dtype=torch.float)
        y_cpu = torch.dot(x1_cpu, x2_cpu)

        x1_xpu = x1_cpu.to("xpu")
        x2_xpu = x2_cpu.to("xpu")
        y_xpu = torch.dot(x1_xpu, x2_xpu)
        print("cpu result:", y_cpu)
        print("xpu result:", y_xpu.cpu())
        self.assertEqual(y_cpu, y_xpu.cpu())

    @pytest.mark.skipif(
        not torch.xpu.has_fp64_dtype(), reason="fp64 not support by this device"
    )
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_dot_double(self, dtype=torch.double):
        x1_cpu = torch.rand(10000, dtype=dtype)
        x2_cpu = torch.rand(10000, dtype=dtype)
        y_cpu = torch.dot(x1_cpu, x2_cpu)

        x1_xpu = x1_cpu.to("xpu")
        x2_xpu = x2_cpu.to("xpu")
        y_xpu = torch.dot(x1_xpu, x2_xpu)
        print("cpu result:", y_cpu)
        print("xpu result:", y_xpu.cpu())
        self.assertEqual(y_cpu, y_xpu.cpu())
