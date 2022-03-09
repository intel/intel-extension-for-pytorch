import time

import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    @pytest.mark.skipif("not torch.xpu.has_onemkl()")
    def test_dot(self, dtype=torch.float):
        x1_cpu = torch.tensor([1.2, 2.4], dtype=torch.float)
        x2_cpu = torch.tensor([3.5, 7.8], dtype=torch.float)
        y_cpu = torch.dot(x1_cpu, x2_cpu)

        x1_xpu = x1_cpu.to("xpu")
        x2_xpu = x2_cpu.to("xpu")
        y_xpu = torch.dot(x1_xpu, x2_xpu)
        self.assertEqual(y_cpu, y_xpu.cpu())
