import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch
import pytest

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_copy_d_to_h_no_contiguous(self, dtype=torch.float):
        input = torch.randn([10000, 64, 3])
        input_xpu = input.to(dpcpp_device)
        output = torch.as_strided(input, (256, 64, 3), (640, 1, 64))
        output_xpu = torch.as_strided(input_xpu, (256, 64, 3), (640, 1, 64))
        self.assertEqual(output, output_xpu.cpu())

    def test_copy_D2D_8d(self, dtype=torch.float):
        input = torch.rand(256, 8, 8, 3, 3, 4, 5, 6, device=cpu_device)
        input_xpu = input.to(dpcpp_device)
        output = input.transpose(3, 4)
        output_xpu = input_xpu.transpose(3, 4)
        output.contiguous()
        output_xpu.contiguous()
        self.assertEqual(output, output_xpu.cpu())
