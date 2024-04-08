import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
        (1, 512, 16, 64)
]

class TestTensorMethod(TestCase):
    def test_clone(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            x_cpu = torch.randn(shape, device=cpu_device)
            x_xpu = x_cpu.to("xpu")
            
            self.assertEqual(x_cpu.clone(), x_xpu.clone().to(cpu_device))
