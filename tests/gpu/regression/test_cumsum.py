import torch
import intel_extension_for_pytorch
import copy
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_cumsum_bool(self):
        input = torch.randint(0, 2, [1, 512], dtype=torch.bool)
        output = torch.empty([1, 512], dtype=torch.int)

        intput_xpu = input.to("xpu")
        output_xpu = torch.empty([1, 512], dtype=torch.int).to("xpu")
        torch.cumsum(input, dim=1, out=output)
        torch.cumsum(intput_xpu, dim=1, out=output_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_cumsum_int(self):
        input = torch.randint(100, [1, 512], dtype=torch.int)
        intput_xpu = input.to("xpu")
        output = torch.cumsum(input, dim=1)
        output_xpu = torch.cumsum(intput_xpu, dim=1)
        self.assertEqual(output, output_xpu.cpu())

    def test_cumprod_bool(self):
        input = torch.randint(0, 2, [1, 512], dtype=torch.bool)
        output = torch.empty([1, 512], dtype=torch.int)

        intput_xpu = input.to("xpu")
        output_xpu = torch.empty([1, 512], dtype=torch.int).to("xpu")
        torch.cumprod(input, dim=1, out=output)
        torch.cumprod(intput_xpu, dim=1, out=output_xpu)
        self.assertEqual(output, output_xpu.cpu())

    def test_cumprod_int(self):
        input = torch.randint(100, [1, 512], dtype=torch.int)
        intput_xpu = input.to("xpu")
        output = torch.cumprod(input, dim=1)
        output_xpu = torch.cumprod(intput_xpu, dim=1)
        self.assertEqual(output, output_xpu.cpu())
    