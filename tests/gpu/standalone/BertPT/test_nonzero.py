import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


class TestNNMethod(TestCase):
    def test_nonzero(self, dtype=torch.float):
        in_cpu = torch.rand((512), dtype=dtype)
        in_xpu = in_cpu.to("xpu")
        
        out_cpu = torch.nonzero(in_cpu)
        out_xpu = torch.nonzero(in_xpu)

        self.assertEqual(out_cpu, out_xpu.cpu())
    
    def test_nonzero_bfloat16(self, dtype=torch.bfloat16):
        in_cpu = torch.rand((512), dtype=dtype)
        in_xpu = in_cpu.to("xpu")
        
        out_cpu = torch.nonzero(in_cpu)
        out_xpu = torch.nonzero(in_xpu)

        self.assertEqual(out_cpu, out_xpu.cpu())

    def test_nonzero_float16(self, dtype=torch.float16):
        in_cpu = torch.rand((512), dtype=dtype)
        in_xpu = in_cpu.to("xpu")
        
        out_cpu = torch.nonzero(in_cpu)
        out_xpu = torch.nonzero(in_xpu)

        self.assertEqual(out_cpu, out_xpu.cpu())
