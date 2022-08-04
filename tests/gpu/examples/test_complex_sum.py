import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_complex_sum(self):
        input = torch.randn([4, 2], dtype=torch.complex64)
        input_xpu = input.to("xpu")
        output = torch.sum(input, dim=-1)
        output_xpu = torch.sum(input_xpu, dim=-1)
        self.assertEqual(output, output_xpu.to("cpu"))
