import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTensorMethod(TestCase):
    def test_rsub(self, dtype=torch.float):
        input_cpu = torch.randn(1, 1, 1, 512)
        input_dpcpp = input_cpu.to(dpcpp_device)

        output_cpu = torch.rsub(input_cpu, 2)
        out_dpcpp = torch.rsub(input_dpcpp, 2)

        # print("input_cpu = ", input_cpu)
        # print("input_dpcpp = ", input_dpcpp)

        self.assertEqual(output_cpu, out_dpcpp)
