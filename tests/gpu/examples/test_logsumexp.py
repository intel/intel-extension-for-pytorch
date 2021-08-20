import torch
from torch.testing._internal.common_utils import TestCase
import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTensorMethod(TestCase):
    def test_logsumexp(self, dtype=torch.float):
        input_cpu = torch.randn(3, 3, dtype=dtype)
        output_cpu = input_cpu.logsumexp(dim=1)

        input_dpcpp = input_cpu.to(dpcpp_device)
        output_dpcpp = input_dpcpp.logsumexp(dim=1)

        print("input_cpu = ", input_cpu)
        print("input_dpcpp = ", input_dpcpp.to(cpu_device))

        self.assertEqual(input_cpu, input_dpcpp.to(cpu_device))
        self.assertEqual(output_cpu, output_dpcpp.to(cpu_device))
