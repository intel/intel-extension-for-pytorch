import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_add(self, dtype=torch.float):
        input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input2 = torch.tensor([-1 + 2j, 1 - 2j, 2 + 1j])
        input1_xpu = input1.to("xpu")
        input2_xpu = input2.to("xpu")
        output = input1 + input2
        output_xpu = input1_xpu + input2_xpu
        self.assertEqual(output, output_xpu.to("cpu"))

    def test_mul(self, dtype=torch.float):
        input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input2 = torch.tensor([-1 + 2j, 1 - 2j, 2 + 1j])
        input1_xpu = input1.to("xpu")
        input2_xpu = input2.to("xpu")
        output = input1 * input2
        output_xpu = input1_xpu * input2_xpu
        self.assertEqual(output, output_xpu.to("cpu"))

    def test_div(self, dtype=torch.float):
        input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input2 = torch.tensor([-1 + 2j, 1 - 2j, 2 + 1j])
        input1_xpu = input1.to("xpu")
        input2_xpu = input2.to("xpu")
        output = input1 / input2
        output_xpu = input1_xpu / input2_xpu
        print(output)
        print(output_xpu.cpu())
        self.assertEqual(output, output_xpu.to("cpu"))
