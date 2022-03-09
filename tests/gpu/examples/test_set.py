import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_set(self, dtype=torch.float):
        x_cpu1 = torch.randn((5, 4))
        x_cpu2 = torch.randn((5, 4))
        x_dpcpp1 = x_cpu1.to("xpu")
        x_dpcpp2 = x_cpu2.to("xpu")

        print("Before:")
        print("self dpcpp", x_dpcpp1.to("cpu"))
        print("src dpcpp", x_dpcpp2.to("cpu"))

        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(x_cpu2, x_dpcpp2.cpu())

        x_cpu1.set_(x_cpu2)
        x_dpcpp1.set_(x_dpcpp2)

        print("After:")
        print("self dpcpp", x_dpcpp1.to("cpu"))
        print("src dpcpp", x_dpcpp2.to("cpu"))
        self.assertEqual(x_cpu1, x_dpcpp1.cpu())
        self.assertEqual(x_cpu2, x_dpcpp2.cpu())
