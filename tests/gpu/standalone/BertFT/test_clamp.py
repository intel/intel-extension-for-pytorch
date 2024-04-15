import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_clamp(self, dtype=torch.float):
        input_cpu = torch.randn(2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)

        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(input_cpu.clamp(min=0, max=0), input_dpcpp.clamp(min=0, max=0))
        self.assertEqual(input_cpu.clamp(min=0, max=2), input_dpcpp.clamp(min=0, max=2))

    def test_clamp_bfloat16(self, dtype=torch.bfloat16):
        input_cpu = torch.randn(2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)

        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(input_cpu.clamp(min=0, max=0), input_dpcpp.clamp(min=0, max=0))
        self.assertEqual(input_cpu.clamp(min=0, max=2), input_dpcpp.clamp(min=0, max=2))

    def test_clamp_float16(self, dtype=torch.float16):
        input_cpu = torch.randn(2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)

        #print("input_cpu = ", input_cpu)
        #print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(input_cpu.clamp(min=0, max=0), input_dpcpp.clamp(min=0, max=0))
        self.assertEqual(input_cpu.clamp(min=0, max=2), input_dpcpp.clamp(min=0, max=2))
