import torch
import intel_extension_for_pytorch

from torch.testing._internal.common_utils import TestCase
import pytest

class TestTorchMethod(TestCase):
    def test_bmm(self):
        batch1 = torch.randn([49152, 27, 3], dtype=torch.complex128)
        batch2 = torch.randn([49152, 3, 1], dtype=torch.complex128)

        output_cpu = torch.bmm(batch1, batch2)
        output_xpu = torch.bmm(batch1.to("xpu"), batch2.to("xpu"))

        self.assertEqual(output_xpu.to("cpu"), output_cpu)