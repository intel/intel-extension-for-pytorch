import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa



class TestTorchMethod(TestCase):
    def test_maximum(self, dtype=torch.float):
        src_1 = torch.randn(2, 4)
        src_2 = torch.randn(2, 4)

        src_gpu_1 = src_1.to("xpu")
        src_gpu_2 = src_2.to("xpu")

        dst = torch.max(src_1, src_2)

        dst_gpu = torch.max(src_gpu_1, src_gpu_2)

        # print("cpu result:", dst)
        # print("gpu result:", dst_gpu.cpu())

        self.assertEqual(dst, dst_gpu)
