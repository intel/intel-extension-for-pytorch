import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

shapes = [
        (1024),
        (1, 384),
        (2, 384),
        (768, 2),
        (384, 1024),
        (768, 1024),
        (768, 4096),
        (2, 384, 1024),
        (2, 384, 4096),
        (2, 384, 16, 64),
        (2, 16, 384, 384)
]

class TestTensorMethod(TestCase):
    def test_view(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            input_cpu = torch.randn(shape)
            input_dpcpp = input_cpu.to(dpcpp_device)

            output_cpu = input_cpu.view(-1, 8)
            out_dpcpp = input_dpcpp.view(-1, 8)

            #print("input_cpu = ", input_cpu)
            #print("input_dpcpp = ", input_dpcpp)

            self.assertEqual(output_cpu, out_dpcpp.cpu())
