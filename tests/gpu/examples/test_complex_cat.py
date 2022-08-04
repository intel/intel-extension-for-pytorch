import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_complex_cat(self, dtype=torch.complex64):
        input1 = torch.randn([4, 2], dtype=dtype)
        input2 = torch.randn([4, 2], dtype=dtype)
        input3 = torch.randn([4, 2], dtype=dtype)
        input1_xpu = input1.to(dpcpp_device)
        input2_xpu = input2.to(dpcpp_device)
        input3_xpu = input3.to(dpcpp_device)
        output = torch.cat([input1, input2, input3], dim=-1)
        output_xpu = torch.cat([input1_xpu, input2_xpu, input3_xpu], dim=-1)
        self.assertEqual(output, output_xpu.to("cpu"))
