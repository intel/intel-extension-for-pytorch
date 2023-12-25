import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch  # noqa

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

shapes_squeeze = [
                (2, 384, 1)
            ]

shapes_unsqueeze = [
                (2, 384),
                (2, 1, 384)
            ]

class TestTorchMethod(TestCase):
    def test_squeeze(self, dtype=torch.float):
        for shape in shapes_squeeze:
            print("\n================== test shape: ", shape, "==================")
            x = torch.randn(shape, device=cpu_device)
            y_cpu = x.squeeze(1)
            print("y = ", y_cpu)

            x_xpu = x.to("xpu")
            y_xpu = x_xpu.squeeze(1)
            print("y_xpu ", y_xpu.cpu())

            self.assertEqual(y_cpu, y_xpu.to(cpu_device))

    def test_unsqueeze(self, dtype=torch.float):
        for shape in shapes_unsqueeze:
            print("\n================== test shape: ", shape, "==================")
            x = torch.randn(shape, device=cpu_device)
            y_cpu = x.unsqueeze(1)
            print("y = ", y_cpu)

            x_xpu = x.to("xpu")
            y_xpu = x_xpu.unsqueeze(1)
            print("y_xpu ", y_xpu.cpu())

            self.assertEqual(y_cpu, y_xpu.to(cpu_device))
