import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_nonzero(self, dtype=torch.float):
        in_cpu = torch.rand((512))
        in_xpu = in_cpu.to("xpu")
        
        out_cpu = torch.nonzero(in_cpu)
        out_xpu = torch.nonzero(in_xpu)

        self.assertEqual(out_cpu, out_xpu.cpu())
