import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch # noqa

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_conj_physical(self, dtype=torch.float):
        input = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input_xpu = input.to("xpu")
        output = torch.conj_physical(input)
        output_xpu = torch.conj_physical(input_xpu)
        self.assertEqual(output, output_xpu.to("cpu"))

        input = torch.tensor([-1., -2., 3.])
        input_xpu = input.to("xpu")
        output = torch.conj_physical(input)
        output_xpu = torch.conj_physical(input_xpu)
        self.assertEqual(output, output_xpu.to("cpu"))

    def test_angle(self, dtype=torch.float):
        input1 = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        input2 = torch.tensor([-1, -2, 3])
        input1_xpu = input1.to("xpu")
        input2_xpu = input2.to("xpu")
        self.assertEqual(torch.angle(input1), torch.angle(input1_xpu).cpu())
        self.assertEqual(torch.angle(input2), torch.angle(input2_xpu).cpu())

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
