import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa


cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes = [
        (2, 384),
        (2, 384, 16, 64)
]

class TestTorchMethod(TestCase):
    def test_contiguouse(self, dtype=torch.float):
        for shape in shapes:
            print("\n================== test shape: ", shape, "==================")
            user_cpu = torch.randn(shape, device=cpu_device)
            res_cpu = user_cpu.contiguous()
            print("begin xpu compute:")
            res_xpu = user_cpu.to("xpu").contiguous()
            print("xpu result:")
            print(res_xpu.cpu())
            self.assertEqual(res_cpu, res_xpu.cpu())
