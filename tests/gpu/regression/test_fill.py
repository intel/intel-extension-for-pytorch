import torch
from torch.testing._internal.common_utils import TestCase
import intel_extension_for_pytorch  # noqa


class TestFill(TestCase):
    def test_fill(self):
        '''
        Regression desc:
          fill_ may set values to part of large-size tensor.
        '''
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

        output_cpu = torch.zeros([2, 8, 256, 512, 224])
        output_xpu = output_cpu.xpu()

        output_cpu.fill_(-1575e-2)
        output_xpu.fill_(-1575e-2)
        self.assertEqual(output_xpu.to("cpu"), output_cpu)

        torch.xpu.empty_cache()
