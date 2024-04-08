import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
        (512)
]

class TestTensorMethod(TestCase):
    def test_ne(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            x_cpu = torch.randn(shape, device=cpu_device)
            x_xpu = x_cpu.to("xpu")
            y_cpu = torch.ne(x_cpu, 2)
            y_xpu = torch.ne(x_xpu, 2)

            self.assertEqual(y_cpu, y_xpu.to(cpu_device))
