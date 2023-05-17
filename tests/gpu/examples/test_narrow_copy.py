import torch
import intel_extension_for_pytorch  # noqa

from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):

    def test_narrow_copy(self, dtype=torch.float):
        dtype = torch.float

        input0 = torch.randn(8192, 8192, device="xpu")
        input0_cpu = input0.to("cpu")
        dim = 0
        start = 4
        end = 4096
        result_out = torch.narrow_copy(input0, dim, start, end)
        result_cpu = torch.narrow_copy(input0_cpu, dim, start, end)
        self.assertEqual(result_out.to("cpu"), result_cpu)

        dim = 1
        start = 1024
        end = 2048
        result_out = torch.narrow_copy(input0, dim, start, end)
        result_cpu = torch.narrow_copy(input0_cpu, dim, start, end)
        self.assertEqual(result_out.to("cpu"), result_cpu)
