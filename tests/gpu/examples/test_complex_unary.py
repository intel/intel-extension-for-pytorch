import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_abs(self, dtype=torch.float):
        input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input2 = input1.to(torch.complex64)
        input1_xpu = input1.to("xpu")
        input2_xpu = input2.to("xpu")
        self.assertEqual(torch.abs(input1), torch.abs(input1_xpu).cpu())
        self.assertEqual(torch.abs(input2), torch.abs(input2_xpu).cpu())

    def test_exp(self, dtype=torch.float):
        input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input2 = input1.to(torch.complex64)
        input1_xpu = input1.to("xpu")
        input2_xpu = input2.to("xpu")
        self.assertEqual(torch.exp(input1), torch.exp(input1_xpu).cpu())
        self.assertEqual(torch.exp(input2), torch.exp(input2_xpu).cpu())
