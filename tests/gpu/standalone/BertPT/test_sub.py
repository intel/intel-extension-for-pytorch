import torch
import pytest
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
        (1, 1, 1, 512)
]

class TestTensorMethod(TestCase):
    def test_sub(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            x_cpu = torch.randn(shape, device=cpu_device)
            x_xpu = x_cpu.to("xpu")
            Other = torch.randn(shape, device=cpu_device)
            Other_xpu = Other.to("xpu")

            self.assertEqual(torch.sub(x_cpu, 1), torch.sub(x_xpu, 1).to(cpu_device))
            self.assertEqual(torch.sub(x_cpu, Other), torch.sub(x_xpu, Other_xpu).to(cpu_device))
