import torch
from torch.testing._internal.common_utils import TestCase

import intel_extension_for_pytorch # noqa

cpu_device = torch.device('cpu')
dpcpp_device = torch.device("xpu")


class TestTorchMethod(TestCase):
    def test_masked_scatter(self, dtype=torch.float):

        x_cpu = torch.rand([2, 3, 4], device=cpu_device)
        y_cpu = torch.rand([2, 3, 4], device=cpu_device)
        mask_cpu = y_cpu.ge(0.5)
        z_cpu = torch.zeros_like(x_cpu)

        z_cpu.masked_scatter_(mask_cpu, x_cpu)
        print("z_cpu:")
        print(z_cpu)

        z_dpcpp = z_cpu.to("xpu")
        z_dpcpp.masked_scatter_(mask_cpu.to("xpu"), x_cpu.to("xpu"))
        print("z_dpcpp:")
        print(z_dpcpp.to("cpu"))
        self.assertEqual(z_cpu, z_dpcpp.cpu())

# For debug
# print("mask_cpu:")
# print(mask_cpu)

# print("y_cpu:")
# print(y_cpu)
