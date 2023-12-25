import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
        (2, 1024),
        (1024, 1024),
        (1024, 4096),
        (4096, 1024),
        (2, 16, 384, 64),
]

class TestTensorMethod(TestCase):
    def test_transpose(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            x_cpu = torch.randn(shape, device=cpu_device)
            x_xpu = x_cpu.to("xpu")
            y_cpu = torch.transpose(x_cpu, 0, 0)
            y_xpu = torch.transpose(x_xpu, 0, 0)

            self.assertEqual(y_cpu, y_xpu.to(cpu_device))
