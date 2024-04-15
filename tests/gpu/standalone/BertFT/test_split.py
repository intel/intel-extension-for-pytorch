import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_split(self, dtype=torch.float):
        input_cpu = torch.arange(2, 384, 2)
        input_dpcpp = input_cpu.to(dpcpp_device)

        # print("input_cpu = ", input_cpu)
        # print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(input_cpu.split(split_size=2), input_dpcpp.split(split_size=2))

    def test_split_bfloat16(self, dtype=torch.bfloat16):
        input_cpu = torch.arange(2, 384, 2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)

        # print("input_cpu = ", input_cpu)
        # print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(input_cpu.split(split_size=2), input_dpcpp.split(split_size=2))

    def test_split_float16(self, dtype=torch.float16):
        input_cpu = torch.arange(2, 384, 2, dtype=dtype)
        input_dpcpp = input_cpu.to(dpcpp_device)

        # print("input_cpu = ", input_cpu)
        # print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(input_cpu.split(split_size=2), input_dpcpp.split(split_size=2))
