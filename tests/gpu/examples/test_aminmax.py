import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa



class TestTorchMethod(TestCase):
    def test_aminmax(self, dtype=torch.float):
        # Test aminmax without dim
        src_cpu = torch.randn(2, 5)
        dst_cpu = torch.aminmax(src_cpu)

        src_gpu = src_cpu.to("xpu")
        dst_gpu = torch.aminmax(src_gpu)

        self.assertEqual(dst_cpu, dst_gpu)

        # Test aminmax with dim
        src_cpu = torch.randn(2, 5)
        dst_cpu = torch.aminmax(src_cpu, dim=0)

        src_gpu = src_cpu.to("xpu")
        dst_gpu = torch.aminmax(src_gpu, dim=0)

        self.assertEqual(dst_cpu, dst_gpu)
